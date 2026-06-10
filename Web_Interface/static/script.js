// =============================================================================
// Three.js scene configuration
// =============================================================================
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.03);

const visualizerContainer = document.getElementById('canvas-container');
const visualizerGroup = new THREE.Group();
scene.add(visualizerGroup);

function getVisualizerBounds() {
    const width = visualizerContainer.clientWidth || window.innerWidth || 1;
    const height = visualizerContainer.clientHeight || 300;
    return { width, height };
}

const initialBounds = getVisualizerBounds();
const camera = new THREE.PerspectiveCamera(75, initialBounds.width / initialBounds.height, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

renderer.setSize(initialBounds.width, initialBounds.height, false);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.domElement.style.touchAction = 'none';
visualizerContainer.appendChild(renderer.domElement);

// =============================================================================
// Central audio-reactive core
// =============================================================================
const geometryRing = new THREE.TorusKnotGeometry(2.9, 0.4, 150, 20);
const materialRing = new THREE.MeshBasicMaterial({
    color: 0x00f3ff,
    wireframe: true,
    transparent: true,
    opacity: 0.5
});
const meshRing = new THREE.Mesh(geometryRing, materialRing);
visualizerGroup.add(meshRing);

const geometryCore = new THREE.TorusKnotGeometry(2.9, 0.35, 150, 20);
const materialCore = new THREE.MeshBasicMaterial({ color: 0x000000 });
const meshCore = new THREE.Mesh(geometryCore, materialCore);
visualizerGroup.add(meshCore);

meshRing.position.y = -0.5;
meshCore.position.y = -0.5;

// =============================================================================
// Radial FFT bars
// =============================================================================
const frequencyBoxes = [];
const numBoxes = 32;
const radius = 4.5;
const boxGeo = new THREE.BoxGeometry(0.15, 0.2, 0.15);
boxGeo.translate(0, 0.1, 0);

const boxMat = new THREE.MeshBasicMaterial({
    color: 0x00f3ff,
    transparent: true,
    opacity: 0.8,
    wireframe: false
});

for (let i = 0; i < numBoxes; i++) {
    const box = new THREE.Mesh(boxGeo, boxMat);
    const angle = (i / numBoxes) * Math.PI * 2;

    box.position.x = Math.cos(angle) * radius;
    box.position.z = Math.sin(angle) * radius;
    box.position.y = -0.5;
    box.lookAt(0, -0.5, 0);

    visualizerGroup.add(box);
    frequencyBoxes.push(box);
}

// =============================================================================
// Scene atmosphere
// =============================================================================
const gridHelper = new THREE.GridHelper(50, 50, 0x222222, 0x111111);
gridHelper.position.y = -5;
scene.add(gridHelper);

const particlesGeo = new THREE.BufferGeometry();
const particlesCount = 1000;
const posArray = new Float32Array(particlesCount * 3);

for (let i = 0; i < particlesCount * 3; i += 3) {
    posArray[i] = (Math.random() - 0.5) * 30;
    posArray[i + 1] = (Math.random() - 0.5) * 30;
    posArray[i + 2] = (Math.random() - 0.5) * 30;
}

particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMat = new THREE.PointsMaterial({
    size: 0.02,
    color: 0xffffff,
    transparent: true,
    opacity: 0.3
});
const particlesMesh = new THREE.Points(particlesGeo, particlesMat);
scene.add(particlesMesh);

function getResponsiveSceneState(width, height) {
    const aspect = width / Math.max(height, 1);
    const shortestSide = Math.min(width, height);
    const compact = width <= 1023;
    const phone = width <= 640;
    const scaleFromSize = THREE.MathUtils.clamp(shortestSide / 600, 0.84, 1.24);
    const scale = compact ? scaleFromSize : 1;
    const cameraZ = phone ? (aspect < 0.75 ? 8.8 : 8.2) : compact ? (aspect < 1 ? 8.1 : 7.7) : 8;
    const cameraY = phone ? -1.15 : compact ? -0.95 : 0;

    return { scale, cameraZ, cameraY };
}

function resizeVisualizer() {
    const { width, height } = getVisualizerBounds();
    const { scale, cameraZ, cameraY } = getResponsiveSceneState(width, height);

    camera.aspect = width / height;
    camera.position.set(0, cameraY, cameraZ);
    camera.lookAt(0, -0.35, 0);
    camera.updateProjectionMatrix();

    visualizerGroup.position.set(0, width <= 1023 ? -1.45 : 0, 0);
    visualizerGroup.scale.setScalar(scale);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
}

function debounce(callback, wait = 120) {
    let timeoutId;
    return (...args) => {
        window.clearTimeout(timeoutId);
        timeoutId = window.setTimeout(() => callback(...args), wait);
    };
}

const debouncedResizeVisualizer = debounce(resizeVisualizer);
resizeVisualizer();

// =============================================================================
// Audio analyzer setup
// =============================================================================
let audioContext;
let analyser;
let dataArray;
let source;
let isAudioInitialized = false;
let currentAudioUrl;

function getAudioElement() {
    return document.getElementById('main-audio');
}

function initAudioSystem() {
    const audioElement = getAudioElement();
    if (!audioElement) return;

    if (!audioContext) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContext();
    }

    if (!isAudioInitialized) {
        try {
            source = audioContext.createMediaElementSource(audioElement);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;

            source.connect(analyser);
            analyser.connect(audioContext.destination);

            dataArray = new Uint8Array(analyser.frequencyBinCount);
            isAudioInitialized = true;
            console.log('Audio analysis pipeline initialized.');
        } catch (error) {
            console.error('Audio analysis pipeline could not be initialized:', error);
        }
    }
}

async function resumeAudioContext() {
    initAudioSystem();

    if (audioContext && audioContext.state === 'suspended') {
        try {
            await audioContext.resume();
        } catch (error) {
            console.error('Audio context resume failed:', error);
        }
    }
}

function attachNativeAudioEvents() {
    const audio = getAudioElement();
    if (!audio) return;

    audio.addEventListener('pointerdown', resumeAudioContext, { passive: true });
    audio.addEventListener('play', async () => {
        await resumeAudioContext();
        updateStatus('Playback started. Frequency analysis is active.');
    });
    audio.addEventListener('pause', () => updateStatus('Playback paused. Output stream is ready.'));
    audio.addEventListener('ended', () => updateStatus('Playback complete. Output stream is ready.'));
    audio.addEventListener('waiting', () => updateStatus('Buffering generated audio stream.'));
    audio.addEventListener('canplay', () => updateStatus('Audio stream loaded. Playback controls are available.'));
    audio.addEventListener('error', () => updateStatus('The browser could not load the generated audio stream.'));
}

// =============================================================================
// Animation loop
// =============================================================================
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const time = clock.getElapsedTime();
    let audioPulse = 1;

    if (isAudioInitialized && analyser && dataArray) {
        analyser.getByteFrequencyData(dataArray);

        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            const freqValue = dataArray[i];
            sum += freqValue;

            if (frequencyBoxes[i]) {
                const targetHeight = 1 + (freqValue / 255) * 15;
                frequencyBoxes[i].scale.y += (targetHeight - frequencyBoxes[i].scale.y) * 0.2;
            }
        }

        const avg = sum / dataArray.length;
        if (avg > 10) {
            audioPulse = 1 + avg / 3000;
        }
    } else {
        for (let i = 0; i < frequencyBoxes.length; i++) {
            frequencyBoxes[i].scale.y += (1 - frequencyBoxes[i].scale.y) * 0.1;
        }
    }

    meshRing.rotation.x = time * 0.1;
    meshRing.rotation.y = time * 0.15;
    meshCore.rotation.x = time * 0.1;
    meshCore.rotation.y = time * 0.15;

    const targetScale = audioPulse;
    meshRing.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    meshCore.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);

    const particleExpansion = 1 + (audioPulse - 1) * 2;
    particlesMesh.scale.set(particleExpansion, particleExpansion, particleExpansion);
    particlesMesh.rotation.y = time * 0.05 * audioPulse;
    gridHelper.position.z = (time * 2) % 1;

    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', debouncedResizeVisualizer, { passive: true });
window.addEventListener('orientationchange', debouncedResizeVisualizer, { passive: true });

// =============================================================================
// UI logic
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    const btnTTS = document.getElementById('btn-tts');
    const btnClone = document.getElementById('btn-clone');
    const dropZone = document.getElementById('drop-zone');
    const fileUpload = document.getElementById('file-upload');
    const speedSlider = document.getElementById('speed-slider');

    attachNativeAudioEvents();
    syncSpeedReadout();

    if (btnTTS) btnTTS.addEventListener('click', (event) => { event.preventDefault(); switchMode('tts'); });
    if (btnClone) btnClone.addEventListener('click', (event) => { event.preventDefault(); switchMode('clone'); });
    if (speedSlider) speedSlider.addEventListener('input', syncSpeedReadout);

    if (dropZone && fileUpload) {
        dropZone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                fileUpload.click();
            }
        });

        dropZone.addEventListener('dragover', (event) => {
            event.preventDefault();
            dropZone.classList.add('drag-active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-active');
        });

        dropZone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropZone.classList.remove('drag-active');

            if (event.dataTransfer.files.length) {
                fileUpload.files = event.dataTransfer.files;
                fileUpload.dispatchEvent(new Event('change'));
            }
        });

        dropZone.addEventListener('touchstart', () => {}, { passive: true });
    }
});

let currentMode = 'tts';

function getSelectedSpeed() {
    const speedSlider = document.getElementById('speed-slider');
    const speed = Number.parseFloat(speedSlider ? speedSlider.value : '1');
    return Number.isFinite(speed) ? speed : 1;
}

function syncSpeedReadout() {
    const speedReadout = document.getElementById('speed-readout');
    if (!speedReadout) return;
    speedReadout.textContent = `${getSelectedSpeed().toFixed(1)}x`;
}

function switchMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;

    document.querySelectorAll('.mode-switch button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${mode}`).classList.add('active');

    const ttsCard = document.getElementById('card-tts');
    const cloneCard = document.getElementById('card-clone');
    const outgoing = mode === 'tts' ? cloneCard : ttsCard;
    const incoming = mode === 'tts' ? ttsCard : cloneCard;

    gsap.to(outgoing, {
        opacity: 0,
        y: 50,
        duration: 0.3,
        onComplete: () => {
            outgoing.classList.add('hidden-card');
            outgoing.classList.remove('active-card');
        }
    });

    setTimeout(() => {
        incoming.classList.remove('hidden-card');
        incoming.classList.add('active-card');
        gsap.fromTo(incoming, { opacity: 0, y: 50 }, { opacity: 1, y: 0, duration: 0.5 });
    }, 300);

    const newColor = mode === 'tts' ? 0x00f3ff : 0xbc13fe;
    const targetColor = new THREE.Color(newColor);

    gsap.to(meshRing.material.color, { r: targetColor.r, g: targetColor.g, b: targetColor.b, duration: 1 });
    gsap.to(boxMat.color, { r: targetColor.r, g: targetColor.g, b: targetColor.b, duration: 1 });
}

// =============================================================================
// Backend connection
// =============================================================================
async function generateTTS() {
    const text = document.getElementById('tts-input').value.trim();
    const speed = getSelectedSpeed();

    if (!text) {
        alert('Enter text before starting synthesis.');
        return;
    }

    await resumeAudioContext();
    updateStatus(`Initializing text segment processing. Speed factor: ${speed.toFixed(1)}x.`);

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, speed })
        });

        if (response.ok) {
            await handleAudioResponse(response);
        } else {
            await handleErrorResponse(response, 'Text-to-speech generation failed.');
        }
    } catch (error) {
        console.error('Text-to-speech request failed:', error);
        updateStatus('Connection failed. Confirm that the Flask server is running.');
    }
}

async function generateClone() {
    const text = document.getElementById('clone-input').value.trim();
    const fileInput = document.getElementById('file-upload');
    const speed = getSelectedSpeed();

    if (!text) {
        alert('Enter target speech text before starting voice cloning.');
        return;
    }

    if (!fileInput.files.length) {
        alert('Upload a reference voice sample before starting voice cloning.');
        return;
    }

    await resumeAudioContext();
    updateStatus('Initializing voice reference analysis. Preparing cloned synthesis request.');

    const formData = new FormData();
    formData.append('text', text);
    formData.append('audio_file', fileInput.files[0]);
    formData.append('speed', speed);

    try {
        const response = await fetch('/api/clone', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            await handleAudioResponse(response);
        } else {
            await handleErrorResponse(response, 'Voice cloning generation failed.');
        }
    } catch (error) {
        console.error('Voice cloning request failed:', error);
        updateStatus('Connection failed. Confirm that the Flask server is running.');
    }
}

async function handleErrorResponse(response, fallbackMessage) {
    try {
        const payload = await response.json();
        updateStatus(payload.error || fallbackMessage);
    } catch {
        updateStatus(fallbackMessage);
    }
}

async function handleAudioResponse(response) {
    const blob = await response.blob();
    const audio = getAudioElement();
    const player = document.getElementById('audio-result');

    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
    }

    currentAudioUrl = URL.createObjectURL(blob);
    audio.src = currentAudioUrl;
    audio.playbackRate = 1;
    audio.load();
    player.classList.add('show');

    try {
        await resumeAudioContext();
        await audio.play();
        updateStatus('Playback started. Frequency analysis is active.');
    } catch (error) {
        console.error('Automatic playback was blocked by the browser:', error);
        updateStatus('Audio stream is ready. Use the browser playback control to start listening.');
    }
}

function updateStatus(message) {
    const statusElement = document.getElementById('status-msg');
    if (statusElement) {
        statusElement.innerText = message;
    }
}

document.getElementById('file-upload').addEventListener('change', function() {
    if (!this.files.length) return;

    const fileName = this.files[0].name;
    const fileStatus = document.getElementById('file-status');
    fileStatus.innerText = `SELECTED: ${fileName}`;
    fileStatus.style.color = '#00f3ff';
});
