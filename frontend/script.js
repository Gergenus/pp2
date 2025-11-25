const API_BASE_URL = 'http://localhost:8000';

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const imagePreview = document.getElementById('imagePreview');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const resultArea = document.getElementById('resultArea');
const resultValue = document.getElementById('resultValue');
const resultDetails = document.getElementById('resultDetails');
const errorDiv = document.getElementById('error');
const fileInfo = document.getElementById('fileInfo');
const riskThreshold = document.getElementById('riskThreshold');

let currentFile = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileInputChange);

    riskThreshold.addEventListener('keypress', handleRiskThresholdKeypress);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave() {
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

function handleFileInputChange(e) {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
}

function handleRiskThresholdKeypress(e) {
    if (e.key === 'Enter' && !processBtn.disabled) {
        processImage();
    }
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Пожалуйста, выберите файл изображения (JPG, PNG)');
        return;
    }

    currentFile = file;
    
    fileInfo.textContent = `Файл: ${file.name} (${formatFileSize(file.size)})`;
    
    processBtn.disabled = false;
    
    showImagePreview(file);
    
    hideResults();
    hideError();
}

function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!currentFile) {
        showError('Пожалуйста, выберите файл');
        return;
    }

    showLoading();
    hideResults();
    hideError();

    try {
        const formData = createFormData();
        const response = await sendImageToServer(formData);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Ошибка сервера');
        }

        showResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function createFormData() {
    const formData = new FormData();
    formData.append('file', currentFile);
    
    const threshold = riskThreshold.value;
    if (threshold && threshold !== '60') {
        formData.append('risk_threshold', threshold);
    }
    
    return formData;
}

async function sendImageToServer(formData) {
    return await fetch(`${API_BASE_URL}/upload-image/`, {
        method: 'POST',
        body: formData
    });
}

function showResults(data) {
    resultValue.textContent = data.result.value.toFixed(2);
    
    const details = `
        <p>📊 Найдено рисков: ${data.result.risks_found}</p>
        <p>🔢 Найдено цифр: ${data.result.digits_found}</p>
        <p>🎯 Порог: ${data.risk_threshold}</p>
        <p>📁 Файл: ${data.filename}</p>
        ${data.result.tip ? `<p>💡 Подсказка: ${data.result.tip}</p>` : ''}
    `;
    resultDetails.innerHTML = details;
    
    resultArea.style.display = 'block';
    resultArea.scrollIntoView({ behavior: 'smooth' });
}

function showLoading() {
    loading.style.display = 'block';
    processBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    processBtn.disabled = false;
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    errorDiv.style.display = 'none';
}

function hideResults() {
    resultArea.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

window.processImage = processImage;