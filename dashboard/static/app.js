let camerasCache = [];
let selectedCameraId = null;


// ==============================
// API
// ==============================

async function get(path) {
    const response = await fetch(path);

    if (!response.ok) {
        throw new Error(await response.text());
    }

    return response.json();
}


async function send(path, method, body) {
    const response = await fetch(path, {
        method,
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(body)
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }

    return data;
}


// ==============================
// HELPERS
// ==============================

function esc(value) {
    return String(value ?? "").replace(
        /[&<>"']/g,
        char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }[char])
    );
}


function openModal(id) {
    document.getElementById(id).classList.add("open");
}


function closeModal(id) {
    document.getElementById(id).classList.remove("open");
}


// ==============================
// CAMERA SELECTION
// ==============================

function selectCamera(id) {

    const camera = camerasCache.find(
        item => Number(item.id) === Number(id)
    );

    if (!camera) {
        return;
    }

    selectedCameraId = Number(id);

    document
        .querySelectorAll(".camera-row")
        .forEach(row => {
            row.classList.toggle(
                "selected",
                Number(row.dataset.cameraId) === selectedCameraId
            );
        });


    const stream = document.getElementById(
        "selected-camera-stream"
    );

    const placeholder = document.getElementById(
        "preview-placeholder"
    );

    const liveBadge = document.getElementById(
        "preview-live-badge"
    );

    const fullscreenButton = document.getElementById(
        "fullscreen-btn"
    );

    const removeButton = document.getElementById(
        "selected-remove-btn"
    );


    document.getElementById(
        "selected-camera-name"
    ).textContent = camera.name;


    document.getElementById(
        "selected-camera-status"
    ).textContent = camera.enabled
        ? "Online / Enabled"
        : "Disabled";


    document.getElementById(
        "preview-camera-id"
    ).textContent =
        `CAM-${String(camera.id).padStart(2, "0")}`;


    document.getElementById(
        "preview-camera-source"
    ).textContent = camera.source;


    const statusDot = document.getElementById(
        "selected-status-dot"
    );

    statusDot.classList.toggle(
        "offline",
        !camera.enabled
    );


    if (camera.enabled) {

        stream.src = `/video_feed/${camera.id}`;
        stream.style.display = "block";

        placeholder.style.display = "none";
        liveBadge.style.display = "flex";

        fullscreenButton.disabled = false;
        removeButton.disabled = false;

    } else {

        stream.removeAttribute("src");
        stream.style.display = "none";

        placeholder.style.display = "flex";
        liveBadge.style.display = "none";

        fullscreenButton.disabled = true;
        removeButton.disabled = false;
    }
}


// ==============================
// FULLSCREEN
// ==============================

function fullscreenCamera() {

    const stream = document.getElementById(
        "selected-camera-stream"
    );

    if (!stream || !stream.src) {
        return;
    }

    if (stream.requestFullscreen) {
        stream.requestFullscreen();
    } else if (stream.webkitRequestFullscreen) {
        stream.webkitRequestFullscreen();
    }
}


// ==============================
// REMOVE SELECTED CAMERA
// ==============================

async function removeSelectedCamera() {

    if (!selectedCameraId) {
        return;
    }

    if (!confirm("Remove this camera?")) {
        return;
    }

    try {

        await send(
            `/api/cameras/${selectedCameraId}`,
            "DELETE"
        );

        selectedCameraId = null;

        stopSelectedCamera();

        await load();

    } catch (error) {

        alert(error.message);
    }
}


function stopSelectedCamera() {

    const stream = document.getElementById(
        "selected-camera-stream"
    );

    stream.removeAttribute("src");
    stream.style.display = "none";

    document.getElementById(
        "preview-placeholder"
    ).style.display = "flex";

    document.getElementById(
        "preview-live-badge"
    ).style.display = "none";

    document.getElementById(
        "selected-camera-name"
    ).textContent = "Select a camera";

    document.getElementById(
        "selected-camera-status"
    ).textContent = "No camera selected";

    document.getElementById(
        "preview-camera-id"
    ).textContent = "—";

    document.getElementById(
        "preview-camera-source"
    ).textContent = "—";

    document.getElementById(
        "fullscreen-btn"
    ).disabled = true;

    document.getElementById(
        "selected-remove-btn"
    ).disabled = true;
}


// ==============================
// CAMERA LIST
// ==============================

function renderCameras(cameras) {

    camerasCache = cameras;

    const container = document.getElementById("cameras");

    const onlineCount = cameras.filter(
        camera => camera.enabled
    ).length;

    document.getElementById(
        "camera-online-count"
    ).textContent = `${onlineCount} online`;

    if (!cameras.length) {

        container.innerHTML = `
            <div class="empty">
                No cameras configured.
            </div>
        `;

        stopSelectedCamera();
        return;
    }

    container.innerHTML = cameras.map(camera => {

        const selected =
            Number(camera.id) === Number(selectedCameraId);

        return `
            <div
                class="camera-row-wrapper
                ${selected ? "selected" : ""}"
                data-camera-id="${camera.id}"
            >

                <button
                    type="button"
                    class="camera-row"
                    onclick="selectCamera(${camera.id})"
                >

                    <div class="camera-row-icon">
                        ◉
                    </div>

                    <div class="camera-row-info">

                        <strong>
                            ${esc(camera.name)}
                        </strong>

                        <span>
                            CAM-${String(camera.id).padStart(2, "0")}
                            ·
                            ${camera.enabled
                                ? "Online"
                                : "Disabled"}
                        </span>

                    </div>

                    <div class="camera-row-status">

                        <span class="camera-status-dot
                            ${camera.enabled
                                ? ""
                                : "offline"}">
                        </span>

                        <span>
                            ${camera.enabled
                                ? "LIVE"
                                : "OFFLINE"}
                        </span>

                    </div>

                    <div class="camera-row-arrow">
                        ›
                    </div>

                </button>


                <!-- REMOVE CAMERA -->

                <button
                    type="button"
                    class="camera-remove-btn"
                    title="Remove camera"
                    onclick="event.stopPropagation();
                             removeCamera(${camera.id})"
                >
                    ×
                </button>

            </div>
        `;

    }).join("");


    // Automatically select first camera
    if (
        selectedCameraId === null &&
        cameras.length > 0
    ) {
        selectCamera(cameras[0].id);
    }

    // Selected camera deleted externally
    else if (
        selectedCameraId !== null &&
        !cameras.some(
            camera =>
                Number(camera.id) ===
                Number(selectedCameraId)
        )
    ) {

        selectedCameraId = null;

        if (cameras.length) {
            selectCamera(cameras[0].id);
        } else {
            stopSelectedCamera();
        }
    }
}


// ==============================
// ADD CAMERA
// ==============================

async function addCamera(event) {

    event.preventDefault();

    try {

        await send(
            "/api/cameras",
            "POST",
            {
                name: document
                    .getElementById("camera-name")
                    .value,

                source: document
                    .getElementById("camera-source")
                    .value
            }
        );

        event.target.reset();

        closeModal("camera-modal");

        await load();

    } catch (error) {

        alert(error.message);
    }
}
async function removeCamera(id) {

    const camera = camerasCache.find(
        item => Number(item.id) === Number(id)
    );

    const name = camera
        ? camera.name
        : `CAM-${id}`;

    if (
        !confirm(
            `Remove "${name}"?\n\nThis will remove the camera from SurvilAI.`
        )
    ) {
        return;
    }

    try {

        await send(
            `/api/cameras/${id}`,
            "DELETE"
        );

        if (
            Number(selectedCameraId) === Number(id)
        ) {
            selectedCameraId = null;
            stopSelectedCamera();
        }

        await load();

    } catch (error) {

        alert(
            `Unable to remove camera:\n${error.message}`
        );
    }
}


// ==============================
// ADD PERSON
// ==============================

async function addPerson(event) {

    event.preventDefault();

    try {

        const name = document
            .getElementById("person-name")
            .value
            .trim();

        const rollNumber = document
            .getElementById("person-roll")
            .value
            .trim();

        if (!name || !rollNumber) {
            alert("Name and Roll Number are required.");
            return;
        }

        await send(
            "/api/people",
            "POST",
            {
                name: name,
                roll_number: rollNumber
            }
        );

        event.target.reset();

        closeModal("person-modal");

        await load();

    } catch (error) {

        alert(error.message);
    }
}


// ==============================
// PEOPLE
// ==============================

function renderPeople(people) {

    const container = document.getElementById("people");

    if (!people.length) {

        container.innerHTML = `
            <tr>
                <td colspan="5" class="empty">
                    No identities enrolled.
                </td>
            </tr>
        `;

        return;
    }

    container.innerHTML = people.map(person => {

        const imageCount = Number(
            person.image_count || 0
        );

        const status = person.active
            ? "Active"
            : "Inactive";

        return `
            <tr>

                <td>
                    <div class="person-table-name">

                        <div class="person-avatar">
                            ${esc(
                                person.name
                                    .charAt(0)
                                    .toUpperCase()
                            )}
                        </div>

                        <strong>
                            ${esc(person.name)}
                        </strong>

                    </div>
                </td>

                <td>
                    ${esc(
                        person.external_ref || "—"
                    )}
                </td>

                <td>
                    <span class="image-count">
                        ${imageCount}
                    </span>
                </td>

                <td>
                    <span class="person-status ${
                        person.active
                            ? "active"
                            : "inactive"
                    }">
                        <span class="person-dot"></span>
                        ${status}
                    </span>
                </td>

                <td>
                    <div class="person-actions">

                        <button
                            class="secondary-btn small-btn"
                            onclick="viewPerson(${person.id})">
                            View
                        </button>

                        <button
                            class="primary-btn small-btn"
                            onclick="addPersonImages(${person.id}, '${esc(person.name)}')">
                            Add
                        </button>

                        <button
                            class="danger small-btn"
                            onclick="deletePerson(${person.id}, '${esc(person.name)}')">
                            Delete
                        </button>

                    </div>
                </td>

            </tr>
        `;

    }).join("");
}


// ==============================
// VIEW PERSON
// ==============================

async function viewPerson(personId) {

    try {

        const person = await send(
            `/api/people/${personId}`,
            "GET"
        );

        alert(
            `Name: ${person.name}\n` +
            `Roll Number: ${person.external_ref || "—"}\n` +
            `Images: ${person.image_count || 0}\n` +
            `Status: ${
                person.active
                    ? "Active"
                    : "Inactive"
            }`
        );

    } catch (error) {

        alert(error.message);
    }
}


// ==============================
// ADD PERSON IMAGES
// ==============================

function addPersonImages(personId, personName) {

    const existing = document.getElementById(
        "person-image-upload-modal"
    );

    if (existing) {
        existing.remove();
    }

    const modal = document.createElement("div");

    modal.id = "person-image-upload-modal";
    modal.className = "modal";

    modal.innerHTML = `
        <form
            class="dialog"
            id="person-image-upload-form"
        >

            <button
                type="button"
                class="close"
                onclick="closePersonImageUpload()"
            >
                ×
            </button>

            <div class="modal-title">
                <span>📸</span>
                <div>
                    <h2>Add Face Images</h2>
                    <p>Enroll images for ${esc(personName)}</p>
                </div>
            </div>

            <label>
                Select Face Images

                <input
                    type="file"
                    id="person-images-input"
                    accept="image/jpeg,image/png,image/webp"
                    multiple
                    required
                >
            </label>

            <div
                id="person-upload-preview"
                class="upload-preview"
            ></div>

            <div
                id="person-upload-status"
                class="upload-status"
            ></div>

            <button
                type="submit"
                class="primary-btn full-btn"
            >
                Upload & Enroll
            </button>

        </form>
    `;

    document.body.appendChild(modal);

    modal.classList.add("open");

    document
        .getElementById("person-images-input")
        .addEventListener(
            "change",
            previewPersonImages
        );

    document
        .getElementById("person-image-upload-form")
        .addEventListener(
            "submit",
            async function(event) {
                event.preventDefault();
                await uploadPersonImages(personId);
            }
        );
}


function closePersonImageUpload() {

    const modal = document.getElementById(
        "person-image-upload-modal"
    );

    if (modal) {
        modal.remove();
    }
}


function previewPersonImages(event) {

    const container = document.getElementById(
        "person-upload-preview"
    );

    if (!container) {
        return;
    }

    container.innerHTML = "";

    Array.from(event.target.files).forEach(file => {

        const item = document.createElement("div");

        item.className = "upload-preview-item";

        const image = document.createElement("img");

        image.src = URL.createObjectURL(file);
        image.alt = file.name;

        item.appendChild(image);
        container.appendChild(item);
    });
}


async function uploadPersonImages(personId) {

    const input = document.getElementById(
        "person-images-input"
    );

    const status = document.getElementById(
        "person-upload-status"
    );

    if (!input || !input.files.length) {

        if (status) {
            status.textContent =
                "Please select at least one image.";
        }

        return;
    }

    const formData = new FormData();

    Array.from(input.files).forEach(file => {

        formData.append("images", file);

    });

    if (status) {
        status.textContent =
            "Detecting faces and creating embeddings...";
    }

    try {

        const response = await fetch(
            `/api/people/${personId}/images`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Image upload failed"
            );
        }

        const added = data.added || [];
        const skipped = data.skipped || [];

        let message =
            `${added.length} image(s) enrolled successfully.`;

        if (skipped.length) {
            message +=
                ` ${skipped.length} image(s) skipped.`;
        }

        if (status) {
            status.textContent = message;
        }

        await load();

        setTimeout(
            closePersonImageUpload,
            1200
        );

    } catch (error) {

        if (status) {
            status.textContent = error.message;
        }

        alert(error.message);
    }
}


// ==============================
// DELETE PERSON
// ==============================

async function deletePerson(personId, personName) {

    const confirmed = confirm(
        `Delete ${personName}?\n\n` +
        `This will also delete all stored face embeddings for this person.`
    );

    if (!confirmed) {
        return;
    }

    try {

        await send(
            `/api/people/${personId}`,
            "DELETE"
        );

        await load();

    } catch (error) {

        alert(error.message);
    }
}


// EVENTS
// ==============================

function renderEvents(events) {

    const container = document.getElementById(
        "events"
    );


    if (!events.length) {

        container.innerHTML = `
            <tr>
                <td colspan="6">
                    No events recorded.
                </td>
            </tr>
        `;

        return;
    }


    container.innerHTML = events.map(event => {

        let snapshot = "No image";


        if (event.snapshot_path) {

            const filename =
                event.snapshot_path
                    .split("/")
                    .pop();

            snapshot = `
                <button
                    class="snapshot-btn"
                    onclick='openSnapshot(
                        ${JSON.stringify(event)}
                    )'
                >
                    View
                </button>
            `;
        }


        return `
            <tr>

                <td>
                    ${esc(event.occurred_at)}
                </td>

                <td>
                    <span class="event">
                        ${esc(event.event_type)}
                    </span>
                </td>

                <td>
                    CAM-${esc(event.camera_id)}
                </td>

                <td>
                    ${esc(event.person_id)}
                </td>

                <td>
                    ${
                        event.confidence == null
                            ? "—"
                            : (
                                Number(event.confidence) * 100
                            ).toFixed(1) + "%"
                    }
                </td>

                <td>
                    ${snapshot}
                </td>

            </tr>
        `;

    }).join("");
}


// ==============================
// SNAPSHOT
// ==============================

function openSnapshot(event) {

    if (!event.snapshot_path) {
        return;
    }


    const filename =
        event.snapshot_path
            .split("/")
            .pop();


    document.getElementById(
        "snapshot-image"
    ).src =
        `/snapshots/${encodeURIComponent(filename)}`;


    document.getElementById(
        "snapshot-info"
    ).innerHTML = `
        <strong>${esc(event.event_type)}</strong>
        <span>
            Camera ${esc(event.camera_id)}
            · Person ${esc(event.person_id)}
            · ${esc(event.occurred_at)}
        </span>
    `;


    openModal("snapshot-modal");
}


// ==============================
// MAIN LOAD
// ==============================

async function load() {

    try {

        const [
            health,
            cameras,
            people,
            events
        ] = await Promise.all([
            get("/api/health"),
            get("/api/cameras"),
            get("/api/people"),
            get("/api/events")
        ]);


        document.getElementById(
            "health"
        ).textContent =
            health.status === "ok"
                ? "● Local / Online"
                : "Offline";


        document.getElementById(
            "camera-count"
        ).textContent = cameras.length;


        document.getElementById(
            "people-count"
        ).textContent = people.length;


        document.getElementById(
            "event-count"
        ).textContent = events.length;


        renderCameras(cameras);
        renderPeople(people);
        renderEvents(events);


    } catch (error) {

        console.error(error);

        document.getElementById(
            "health"
        ).textContent =
            "● Dashboard error";


        document.getElementById(
            "events"
        ).innerHTML = `
            <tr>
                <td colspan="6">
                    Failed to fetch
                </td>
            </tr>
        `;
    }
}


// ==============================
// START
// ==============================

load();

setInterval(load, 5000);