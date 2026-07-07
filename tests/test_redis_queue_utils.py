"""Tests for the Redis frame-queue helpers (run against fakeredis)."""

import numpy as np
import pytest

import facebin.server.redis_queue_utils as rqu
from facebin.config import RedisConfig
from facebin.errors import RedisConnectionError


def make_image(w=4, h=3):
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_connection_error_is_meaningful():
    rqu.set_redis(None)
    rqu.configure(RedisConfig(host="localhost", port=1, db=0))
    try:
        with pytest.raises(RedisConnectionError) as e:
            rqu.get_redis()
        message = str(e.value)
        assert "localhost:1" in message
        assert "redis-server" in message
    finally:
        rqu.configure(RedisConfig())


def test_init_frame_and_queue(fake_redis):
    image = make_image()
    rqu.init_frame(image, 123.0, "video.mp4", "camera1")

    assert rqu.queue_length(rqu.CAMERA_QUEUE) == 1
    key, score = rqu.get_next_key(rqu.CAMERA_QUEUE)
    assert key == "frame:camera1:123.0"
    assert score == 123.0
    # Peek does not remove the key.
    assert rqu.queue_length(rqu.CAMERA_QUEUE) == 1


def test_get_next_key_delete(fake_redis):
    image = make_image()
    rqu.init_frame(image, 1.0, None, "cam")
    key, score = rqu.get_next_key(rqu.CAMERA_QUEUE, delete=True)
    assert key is not None
    assert rqu.queue_length(rqu.CAMERA_QUEUE) == 0
    # Now the queue is empty.
    assert rqu.get_next_key(rqu.CAMERA_QUEUE, delete=True) == (None, None)


def test_get_next_key_empty(fake_redis):
    assert rqu.get_next_key("emptyqueue") == (None, None)


def test_frame_image_roundtrip(fake_redis):
    image = make_image(w=8, h=5)
    key = rqu.add_frame(image, 5.5, "v.mp4", "cam2")
    restored = rqu.get_frame_image(key)
    assert restored is not None
    assert restored.shape == image.shape
    assert restored.dtype == image.dtype
    np.testing.assert_array_equal(restored, image)


def test_get_frame_image_missing(fake_redis):
    assert rqu.get_frame_image(None) is None
    assert rqu.get_frame_image("frame:none:1") is None


def test_get_frame_fields(fake_redis):
    image = make_image()
    key = rqu.add_frame(image, 9.0, "v.mp4", "cam3")
    fields = rqu.get_frame(key, fields=["camera_id", "filename"])
    assert fields["camera_id"] == b"cam3"
    assert fields["filename"] == b"v.mp4"
    everything = rqu.get_frame(key)
    assert "image_data" in everything


def test_frame_expiration_set(fake_redis):
    image = make_image()
    rqu.init_frame(image, 2.0, None, "cam")
    key, _ = rqu.get_next_key(rqu.CAMERA_QUEUE)
    ttl = fake_redis.ttl(key)
    assert 0 < ttl <= rqu.STANDARD_EXPIRATION


def test_del_frame(fake_redis):
    image = make_image()
    key = rqu.add_frame(image, 3.0, None, "cam")
    assert fake_redis.exists(key)
    rqu.del_frame(key)
    assert not fake_redis.exists(key)


def test_make_array_roundtrip():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data, dtype, shape = rqu.make_string(arr)
    restored = rqu.make_array(data, dtype, shape)
    np.testing.assert_array_equal(restored, arr)


def test_recognizer_queue_name():
    assert rqu.RECOGNIZER_QUEUE("cam1") == "recognizer:cam1"
    assert rqu.RECOGNIZER_QUEUE(b"cam1") == "recognizer:cam1"


def test_fix_keys():
    fixed = rqu.fix_keys({b"a": 1, "b": 2})
    assert fixed == {"a": 1, "b": 2}


def test_hash_key_helpers_are_consistent():
    assert rqu.face_x_k(3) == "face_3_x"
    assert rqu.face_image_data_k(0) == "face_image_0_data"
    assert rqu.face_encoding_k(1) == "face_1_encoding"
