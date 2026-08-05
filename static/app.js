/**
 * VeriFact - Frontend JavaScript
 * 
 * Features:
 * - Theme toggle (dark/light)
 * - Loading stages with progress
 * - Copy results to clipboard
 * - Export as JSON
 * - Enhanced evidence cards with source weight
 */

// Theme Toggle
const themeToggle = document.getElementById('theme-toggle-btn');
const htmlElement = document.documentElement;

// Check for saved theme preference or default to 'dark'
const currentTheme = localStorage.getItem('theme') || 'dark';
htmlElement.setAttribute('data-theme', currentTheme);
themeToggle.checked = currentTheme === 'dark';

themeToggle.addEventListener('change', function () {
    if (this.checked) {
        htmlElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    } else {
        htmlElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
});

// Tab switching
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Max results slider
const maxResultsSlider = document.getElementById('max-results');
const maxResultsValue = document.getElementById('max-results-value');

function updateSlider() {
    const value = maxResultsSlider.value;
    const max = maxResultsSlider.max;
    const percentage = (value / max) * 100;

    maxResultsValue.textContent = value;
    maxResultsSlider.style.background = `linear-gradient(to right, #6366f1 0%, #ec4899 ${percentage}%, rgba(148, 163, 184, 0.3) ${percentage}%)`;
}

maxResultsSlider.addEventListener('input', updateSlider);
updateSlider();

// Sections
const inputSection = document.querySelector('.input-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

// Loading stages
const loadingStages = [
    "Extracting claim...",
    "Generating search queries...",
    "Searching for evidence...",
    "Analyzing sources...",
    "Computing verdict..."
];

let loadingInterval = null;
let currentStage = 0;

function startLoadingAnimation() {
    currentStage = 0;
    updateLoadingText();
    loadingInterval = setInterval(() => {
        currentStage = (currentStage + 1) % loadingStages.length;
        updateLoadingText();
    }, 3000);
}

function stopLoadingAnimation() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

function updateLoadingText() {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) {
        loadingText.textContent = loadingStages[currentStage];
    }
}

function showSection(section) {
    [inputSection, loadingSection, resultsSection, errorSection].forEach(s => {
        s.style.display = 'none';
    });
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check button
const checkBtn = document.getElementById('check-btn');
const claimInput = document.getElementById('claim-input');
const urlInput = document.getElementById('url-input');

checkBtn.addEventListener('click', async () => {
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    const claim = activeTab === 'text' ? claimInput.value.trim() : '';
    const url = activeTab === 'url' ? urlInput.value.trim() : '';
    const maxResults = parseInt(maxResultsSlider.value);

    if (!claim && !url) {
        alert('Please enter a claim or URL to fact-check');
        return;
    }

    await checkClaim(claim, url, maxResults);
});

// Allow Enter key in textarea
claimInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        checkBtn.click();
    }
});

// Store last result for export
let lastResult = null;

// API call
async function checkClaim(claim, url, maxResults) {
    showSection(loadingSection);
    startLoadingAnimation();

    const requestBody = {
        max_results: maxResults
    };

    if (claim) {
        requestBody.claim = claim;
    } else if (url) {
        requestBody.url = url;
    }

    try {
        const response = await fetch('/api/check', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        stopLoadingAnimation();

        if (response.status === 429) {
            showError('Rate limit exceeded. Please wait a moment and try again.');
            return;
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'error') {
            showError(data.error || 'An error occurred while processing your request');
        } else {
            lastResult = data;
            displayResults(data);
        }
    } catch (error) {
        stopLoadingAnimation();
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the API. Please try again.');
    }
}

// Display results
function displayResults(data) {
    // Verdict badge
    const verdictBadge = document.getElementById('verdict-badge');
    const verdict = data.verdict.toLowerCase().replace(/[\s_\/]+/g, '-');
    verdictBadge.className = `verdict-badge ${verdict}`;
    verdictBadge.textContent = data.verdict.replace(/_/g, ' ');

    // Claim text
    document.getElementById('claim-text').textContent = data.claim;

    // Confidence
    const confidence = Math.round(data.confidence * 100);
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');

    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    confidenceValue.textContent = `${confidence}%`;

    // Net score
    document.getElementById('net-score').textContent = data.net_score.toFixed(2);

    // Processing time
    document.getElementById('processing-time').textContent = `${data.processing_time}s`;

    // AI Summary section
    const summaryContainer = document.getElementById('ai-summary-container');
    const summaryText = document.getElementById('ai-summary-text');
    if (data.summary && data.summary.trim()) {
        summaryText.textContent = data.summary;
        summaryContainer.style.display = 'block';
    } else {
        summaryContainer.style.display = 'none';
    }

    // Explanation section
    displayExplanation(data.explanation);

    // Evidences
    const evidencesList = document.getElementById('evidences-list');
    evidencesList.innerHTML = '';

    if (data.evidences && data.evidences.length > 0) {
        data.evidences.forEach((evidence, index) => {
            const card = createEvidenceCard(evidence, index + 1);
            evidencesList.appendChild(card);
        });
    } else {
        evidencesList.innerHTML = '<p style="text-align: center; color: #6b7280;">No evidence sources found.</p>';
    }

    showSection(resultsSection);
}

// Display explanation timeline
function displayExplanation(explanation) {
    let container = document.getElementById('explanation-container');

    // Create container if it doesn't exist
    if (!container) {
        const resultsSection = document.getElementById('results-section');
        const verdictCard = resultsSection.querySelector('.verdict-card');
        container = document.createElement('div');
        container.id = 'explanation-container';
        container.className = 'explanation-container glass-card';
        verdictCard.parentNode.insertBefore(container, verdictCard.nextSibling);
    }

    if (!explanation) {
        container.innerHTML = '';
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    // Build steps HTML
    const stepsHtml = explanation.steps.map(step => `
        <div class="explanation-step">
            <div class="step-icon">${step.icon}</div>
            <div class="step-content">
                <div class="step-title">${step.title}</div>
                <div class="step-detail">${step.detail}</div>
            </div>
        </div>
    `).join('<div class="step-connector"></div>');

    // Build breakdown HTML
    const breakdown = explanation.breakdown;
    const total = breakdown.support_count + breakdown.refute_count + breakdown.neutral_count;
    const supportPct = total > 0 ? (breakdown.support_count / total * 100) : 0;
    const refutePct = total > 0 ? (breakdown.refute_count / total * 100) : 0;
    const neutralPct = total > 0 ? (breakdown.neutral_count / total * 100) : 0;

    container.innerHTML = `
        <div class="explanation-header" onclick="toggleExplanation()">
            <span class="explanation-title">🧠 Why this verdict?</span>
            <span class="explanation-toggle" id="explanation-toggle">▼</span>
        </div>
        <div class="explanation-content" id="explanation-content">
            <div class="explanation-timeline">
                ${stepsHtml}
            </div>
            <div class="explanation-breakdown">
                <div class="breakdown-title">Evidence Breakdown</div>
                <div class="breakdown-bar">
                    <div class="breakdown-segment support" style="width: ${supportPct}%" title="${breakdown.support_count} supporting"></div>
                    <div class="breakdown-segment refute" style="width: ${refutePct}%" title="${breakdown.refute_count} refuting"></div>
                    <div class="breakdown-segment neutral" style="width: ${neutralPct}%" title="${breakdown.neutral_count} neutral"></div>
                </div>
                <div class="breakdown-labels">
                    <span class="support-label">✓ ${breakdown.support_count} Support</span>
                    <span class="refute-label">✗ ${breakdown.refute_count} Refute</span>
                    <span class="neutral-label">○ ${breakdown.neutral_count} Neutral</span>
                </div>
                <div class="threshold-info">${explanation.threshold_info}</div>
            </div>
        </div>
    `;
}

// Toggle explanation visibility
function toggleExplanation() {
    const content = document.getElementById('explanation-content');
    const toggle = document.getElementById('explanation-toggle');

    if (content.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
        toggle.textContent = '▼';
    } else {
        content.classList.add('collapsed');
        toggle.textContent = '▶';
    }
}

// Create evidence card with source weight
function createEvidenceCard(evidence, index) {
    const card = document.createElement('div');
    card.className = 'evidence-card glass-card';

    const stance = evidence.stance.toLowerCase();
    const stanceClass = stance === 'supports' ? 'supports' : stance === 'refutes' ? 'refutes' : 'neutral';

    const similarity = Math.round(evidence.similarity * 100);
    const stanceScore = Math.round(evidence.stance_score * 100);
    const sourceWeight = evidence.source_weight ? evidence.source_weight.toFixed(1) : '1.0';
    const isSocialMedia = evidence.is_social_media;

    // Source weight indicator
    let weightBadge = '';
    if (sourceWeight > 1.0) {
        weightBadge = '<span class="weight-badge trusted">🏆 Trusted</span>';
    } else if (sourceWeight < 1.0) {
        weightBadge = '<span class="weight-badge low">⚠️ Low Weight</span>';
    }

    // Social media badge
    const socialBadge = isSocialMedia ? '<span class="social-badge">📱 Social</span>' : '';

    card.innerHTML = `
        <div class="evidence-header">
            <a href="${evidence.url}" target="_blank" rel="noopener noreferrer" class="evidence-url">
                Source ${index}: ${new URL(evidence.url).hostname}
            </a>
            <div class="badges">
                ${socialBadge}
                ${weightBadge}
                <span class="stance-badge ${stanceClass}">${evidence.stance}</span>
            </div>
        </div>
        <p class="evidence-text">${evidence.best_sentence}</p>
        <div class="evidence-scores">
            <div>Similarity: <span>${similarity}%</span></div>
            <div>Stance: <span>${stanceScore}%</span></div>
            <div>Weight: <span>${sourceWeight}x</span></div>
        </div>
    `;

    return card;
}

// Copy results to clipboard
function copyResults() {
    if (!lastResult) return;

    const text = `
Claim: ${lastResult.claim}
Verdict: ${lastResult.verdict}
Confidence: ${Math.round(lastResult.confidence * 100)}%
Net Score: ${lastResult.net_score}
${lastResult.summary ? `\nAI Summary: ${lastResult.summary}` : ''}

Evidence Sources:
${lastResult.evidences.map((e, i) => `${i + 1}. ${e.url}\n   "${e.best_sentence}"\n   Stance: ${e.stance} (${Math.round(e.stance_score * 100)}%)`).join('\n\n')}
    `.trim();

    navigator.clipboard.writeText(text).then(() => {
        alert('Results copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

// Export as JSON
function exportJSON() {
    if (!lastResult) return;

    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fact-check-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Export as PDF using print dialog
function exportPDF() {
    if (!lastResult) {
        alert('No results to export. Please run a fact-check first.');
        return;
    }

    // Verdict styling
    let verdictColor = '#64748b';
    let verdictBg = '#f1f5f9';
    if (lastResult.verdict.includes('TRUE')) {
        verdictColor = '#059669';
        verdictBg = '#d1fae5';
    } else if (lastResult.verdict.includes('FALSE')) {
        verdictColor = '#dc2626';
        verdictBg = '#fee2e2';
    } else if (lastResult.verdict.includes('MIXED')) {
        verdictColor = '#d97706';
        verdictBg = '#fef3c7';
    }

    // Build evidence HTML
    const evidenceHtml = lastResult.evidences.map((e, i) => {
        let stanceColor = '#64748b';
        let stanceBg = '#f1f5f9';
        if (e.stance === 'supports') {
            stanceColor = '#059669';
            stanceBg = '#d1fae5';
        } else if (e.stance === 'refutes') {
            stanceColor = '#dc2626';
            stanceBg = '#fee2e2';
        }

        let hostname = 'Unknown';
        try { hostname = new URL(e.url).hostname; } catch (err) { hostname = e.url.substring(0, 30); }

        return `
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid ${stanceColor};">
                <div style="margin-bottom: 8px;">
                    <strong style="color: #0f172a; font-size: 13px;">Source ${i + 1}: ${hostname}</strong>
                    <span style="background: ${stanceBg}; color: ${stanceColor}; padding: 2px 8px; border-radius: 10px; font-size: 10px; text-transform: uppercase; margin-left: 8px; font-weight: 600;">${e.stance}</span>
                </div>
                <p style="margin: 0 0 6px 0; font-style: italic; color: #475569; font-size: 12px; line-height: 1.5;">"${e.best_sentence}"</p>
                <div style="font-size: 10px; color: #94a3b8;">
                    Similarity: ${Math.round(e.similarity * 100)}% | Stance: ${Math.round(e.stance_score * 100)}% | Weight: ${(e.source_weight || 1.0).toFixed(1)}x
                </div>
            </div>
        `;
    }).join('');

    // Create print window
    const printWindow = window.open('', '_blank', 'width=800,height=600');

    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fact-Check Report</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: Arial, sans-serif; padding: 30px; color: #1f2937; max-width: 800px; margin: 0 auto; }
                @media print { body { padding: 20px; } }
            </style>
        </head>
        <body>
            <div style="text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #e2e8f0;">
                <h1 style="color: #0f172a; margin: 0 0 4px 0; font-size: 22px; font-weight: 700;">Fact-Check Report</h1>
                <p style="color: #64748b; font-size: 11px;">Generated: ${new Date().toLocaleString()}</p>
            </div>
            
            <div style="background: ${verdictBg}; color: ${verdictColor}; padding: 12px 20px; border-radius: 6px; text-align: center; margin-bottom: 20px;">
                <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Verdict</span>
                <h2 style="margin: 4px 0 0 0; font-size: 20px; font-weight: 700;">${lastResult.verdict}</h2>
            </div>
            
            <div style="background: #f8fafc; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                <p style="margin: 0 0 6px 0; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Claim Analyzed</p>
                <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #0f172a;">${lastResult.claim}</p>
            </div>
            
            ${lastResult.summary ? `
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(139, 92, 246, 0.06)); padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 3px solid #6366f1;">
                <p style="margin: 0 0 6px 0; color: #6366f1; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">🤖 AI Summary</p>
                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #0f172a;">${lastResult.summary}</p>
            </div>
            ` : ''}
            
            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                <div style="flex: 1; background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                    <p style="margin: 0; color: #64748b; font-size: 10px; text-transform: uppercase;">Confidence</p>
                    <p style="margin: 4px 0 0 0; font-size: 18px; font-weight: 700; color: #0f172a;">${Math.round(lastResult.confidence * 100)}%</p>
                </div>
                <div style="flex: 1; background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                    <p style="margin: 0; color: #64748b; font-size: 10px; text-transform: uppercase;">Net Score</p>
                    <p style="margin: 4px 0 0 0; font-size: 18px; font-weight: 700; color: #0f172a;">${lastResult.net_score.toFixed(2)}</p>
                </div>
                <div style="flex: 1; background: #f8fafc; padding: 12px; border-radius: 6px; text-align: center;">
                    <p style="margin: 0; color: #64748b; font-size: 10px; text-transform: uppercase;">Time</p>
                    <p style="margin: 4px 0 0 0; font-size: 18px; font-weight: 700; color: #0f172a;">${lastResult.processing_time}s</p>
                </div>
            </div>
            
            <div style="margin-bottom: 10px;">
                <p style="color: #0f172a; font-size: 14px; font-weight: 600; margin: 0 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;">
                    Evidence Sources (${lastResult.evidences.length})
                </p>
                ${evidenceHtml}
            </div>
            
            <div style="text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                <p style="color: #94a3b8; font-size: 10px;">VeriFact • SBERT + DeBERTa-v3</p>
            </div>
            
            <script>
                window.onload = function() {
                    window.print();
                };
            </script>
        </body>
        </html>
    `);

    printWindow.document.close();
}

// Show error
function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection(errorSection);
}

// Retry button
document.getElementById('retry-btn').addEventListener('click', () => {
    showSection(inputSection);
});

// Check another button
document.getElementById('check-another-btn').addEventListener('click', () => {
    claimInput.value = '';
    urlInput.value = '';
    lastResult = null;
    showSection(inputSection);
});

// Add copy, export, and PDF buttons dynamically
document.addEventListener('DOMContentLoaded', () => {
    const checkAnotherBtn = document.getElementById('check-another-btn');
    if (checkAnotherBtn) {
        // Create a container for all buttons
        const buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.flexDirection = 'column';
        buttonContainer.style.alignItems = 'center';
        buttonContainer.style.gap = '16px';
        buttonContainer.style.marginTop = '32px';

        // Create row for export buttons
        const exportRow = document.createElement('div');
        exportRow.style.display = 'flex';
        exportRow.style.justifyContent = 'center';
        exportRow.style.gap = '12px';
        exportRow.style.flexWrap = 'wrap';

        // Add copy button
        const copyBtn = document.createElement('button');
        copyBtn.id = 'copy-results-btn';
        copyBtn.className = 'secondary-btn';
        copyBtn.textContent = '📋 Copy Results';
        copyBtn.onclick = copyResults;

        // Add export button
        const exportBtn = document.createElement('button');
        exportBtn.id = 'export-json-btn';
        exportBtn.className = 'secondary-btn';
        exportBtn.textContent = '💾 Export JSON';
        exportBtn.onclick = exportJSON;

        // Add PDF button
        const pdfBtn = document.createElement('button');
        pdfBtn.id = 'export-pdf-btn';
        pdfBtn.className = 'secondary-btn';
        pdfBtn.textContent = '📄 Export PDF';
        pdfBtn.onclick = exportPDF;

        // Add buttons to export row
        exportRow.appendChild(copyBtn);
        exportRow.appendChild(exportBtn);
        exportRow.appendChild(pdfBtn);

        // Move check another btn to container
        const parentNode = checkAnotherBtn.parentNode;
        parentNode.removeChild(checkAnotherBtn);

        buttonContainer.appendChild(exportRow);
        buttonContainer.appendChild(checkAnotherBtn);

        parentNode.appendChild(buttonContainer);
    }
});
