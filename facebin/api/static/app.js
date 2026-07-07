/* Facebin mobile web app.
 *
 * A small vanilla-JS single-page app over the /api endpoints: login,
 * live MJPEG camera streams, appearance history, people, and pipeline
 * status. The bearer token is kept in localStorage; stream <img> tags
 * receive it as a ?token= parameter because they cannot send headers.
 */

"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  token: localStorage.getItem("facebin-token"),
  tab: "live",
};

/* ---- API helpers ---- */

async function api(path, options = {}) {
  const headers = Object.assign(
    { Authorization: "Bearer " + state.token },
    options.headers || {});
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, Object.assign({}, options, { headers }));
  if (response.status === 401) {
    setToken(null);
    showLogin();
    throw new Error("Session expired; please sign in again.");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* keep */ }
    throw new Error(detail);
  }
  return response.json();
}

function setToken(token) {
  state.token = token;
  if (token) localStorage.setItem("facebin-token", token);
  else localStorage.removeItem("facebin-token");
}

/* ---- Views ---- */

function showLogin() {
  $("#login-view").hidden = false;
  $("#app-view").hidden = true;
  stopStreams();
}

function showApp() {
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  selectTab(state.tab);
}

function selectTab(name) {
  state.tab = name;
  for (const button of document.querySelectorAll("nav button")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.hidden = tab.id !== "tab-" + name;
  }
  $("#page-title").textContent =
    { live: "Live", history: "History", people: "People", status: "Status" }[name];
  if (name !== "live") stopStreams();
  const loaders = { live: loadLive, history: loadHistory, people: loadPeople,
                    status: loadStatus };
  loaders[name]().catch((e) => console.error(e));
}

/* ---- Live ---- */

function stopStreams() {
  for (const img of document.querySelectorAll("#camera-list img")) {
    img.src = "";  // closes the MJPEG connection
  }
}

async function loadLive() {
  const cameras = await api("/api/cameras");
  const list = $("#camera-list");
  list.innerHTML = "";
  $("#live-empty").hidden = cameras.length > 0;
  for (const cam of cameras) {
    const card = document.createElement("div");
    card.className = "camera-card";
    const img = document.createElement("img");
    img.alt = "Live stream of " + cam.name;
    img.src = cam.stream_url + "?token=" + encodeURIComponent(state.token);
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = cam.name;
    card.append(img, name);
    list.append(card);
  }
}

/* ---- History ---- */

function formatTime(ts) {
  return new Date(ts * 1000).toLocaleString([], {
    dateStyle: "short", timeStyle: "medium" });
}

async function loadHistory() {
  const unknownOnly = $("#history-filter").value === "unknown";
  const records = await api("/api/history?limit=100" +
                            (unknownOnly ? "&unknown_only=true" : ""));
  const list = $("#history-list");
  list.innerHTML = "";
  $("#history-empty").hidden = records.length > 0;
  for (const rec of records) {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = rec.face_image_url + "?token=" + encodeURIComponent(state.token);
    img.onerror = () => { img.replaceWith(placeholder("?")); };
    const info = document.createElement("div");
    info.className = "info";
    info.innerHTML =
      '<div class="primary"></div><div class="secondary"></div>';
    info.querySelector(".primary").textContent =
      rec.known ? rec.person_name || "(unnamed)" : "Unknown person";
    info.querySelector(".secondary").textContent =
      formatTime(rec.time) + " · camera " + rec.camera_id;
    const badge = document.createElement("span");
    badge.className = "badge" + (rec.known ? "" : " unknown");
    badge.textContent = rec.known ? "KNOWN" : "UNKNOWN";
    li.append(img, info, badge);
    list.append(li);
  }
}

function placeholder(text) {
  const div = document.createElement("div");
  div.className = "placeholder";
  div.textContent = text;
  return div;
}

/* ---- People ---- */

async function loadPeople() {
  const people = await api("/api/persons");
  const list = $("#person-list");
  list.innerHTML = "";
  $("#people-empty").hidden = people.length > 0;
  for (const person of people) {
    const li = document.createElement("li");
    if (person.image_url) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.src = person.image_url + "?token=" + encodeURIComponent(state.token);
      img.onerror = () => { img.replaceWith(placeholder("👤")); };
      li.append(img);
    } else {
      li.append(placeholder("👤"));
    }
    const info = document.createElement("div");
    info.className = "info";
    info.innerHTML =
      '<div class="primary"></div><div class="secondary"></div>';
    info.querySelector(".primary").textContent = person.name;
    info.querySelector(".secondary").textContent =
      [person.title, person.face_count + " face image(s)"]
        .filter(Boolean).join(" · ");
    li.append(info);
    list.append(li);
  }
}

/* ---- Status ---- */

async function loadStatus() {
  const status = await api("/api/status");
  const lines = ["version   " + status.version,
                 "database  " + status.database, ""];
  for (const [queue, length] of Object.entries(status.queues)) {
    lines.push(queue.padEnd(24, " ") + length);
  }
  $("#status-body").textContent = lines.join("\n");
}

/* ---- Wiring ---- */

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorBox = $("#login-error");
  errorBox.hidden = true;
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: $("#login-username").value,
        password: $("#login-password").value,
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Login failed.");
    }
    setToken((await response.json()).token);
    showApp();
  } catch (e) {
    errorBox.textContent = e.message;
    errorBox.hidden = false;
  }
});

$("#logout-button").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch (e) { /* ok */ }
  setToken(null);
  showLogin();
});

$("#person-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorBox = $("#person-error");
  errorBox.hidden = true;
  try {
    await api("/api/persons", {
      method: "POST",
      body: JSON.stringify({
        name: $("#person-name").value,
        title: $("#person-title").value,
        notes: $("#person-notes").value,
      }),
    });
    $("#person-form").reset();
    await loadPeople();
  } catch (e) {
    errorBox.textContent = e.message;
    errorBox.hidden = false;
  }
});

for (const button of document.querySelectorAll("nav button")) {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
}

$("#history-filter").addEventListener("change", () => loadHistory());
$("#history-refresh").addEventListener("click", () => loadHistory());

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => { /* offline is best-effort */ });
}

if (state.token) {
  api("/api/me").then(showApp).catch(showLogin);
} else {
  showLogin();
}
