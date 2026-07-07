/* Facebin service worker: cache the app shell so the UI opens instantly;
 * everything under /api always goes to the network (live data). */

"use strict";

const CACHE = "facebin-shell-v1";
const SHELL = ["./", "index.html", "style.css", "app.js",
               "manifest.webmanifest", "icons/icon-192.png",
               "icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE)
                      .map((k) => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/") || event.request.method !== "GET") {
    return; // live data: straight to the network
  }
  event.respondWith(
    caches.match(event.request).then(
      (cached) => cached || fetch(event.request)));
});
