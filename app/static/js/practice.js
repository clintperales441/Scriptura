/**
 * Scriptura — Practice Page Interactivity
 * Handles: countdown timer, word count, auto-save to sessionStorage
 */

(function () {
    'use strict';

    // --- DOM Elements ---
    const timerDisplay = document.getElementById('timer');
    const writingInput = document.getElementById('writing-input');
    const wordCountDisplay = document.getElementById('word-count');
    const writingForm = document.getElementById('writing-form');
    const finishBtn = document.getElementById('finish-btn');

    if (!timerDisplay || !writingInput || !wordCountDisplay) return;

    // --- Configuration ---
    const totalSeconds = (typeof TIMER_MINUTES !== 'undefined' ? TIMER_MINUTES : 5) * 60;
    const sessionId = typeof SESSION_ID !== 'undefined' ? SESSION_ID : 'unknown';
    const storageKey = `scriptura_draft_${sessionId}`;

    // --- State ---
    let remainingSeconds = totalSeconds;
    let timerInterval = null;

    // --- Word Count ---
    function countWords(text) {
        const trimmed = text.trim();
        if (!trimmed) return 0;
        return trimmed.split(/\s+/).length;
    }

    function updateWordCount() {
        wordCountDisplay.textContent = countWords(writingInput.value);
    }

    // --- Auto-save ---
    function saveDraft() {
        try {
            sessionStorage.setItem(storageKey, writingInput.value);
        } catch (e) {
            // sessionStorage may be unavailable; ignore silently
        }
    }

    function restoreDraft() {
        try {
            const saved = sessionStorage.getItem(storageKey);
            if (saved && !writingInput.value) {
                writingInput.value = saved;
                updateWordCount();
            }
        } catch (e) {
            // Ignore
        }
    }

    function clearDraft() {
        try {
            sessionStorage.removeItem(storageKey);
        } catch (e) {
            // Ignore
        }
    }

    // --- Timer ---
    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function updateTimerDisplay() {
        timerDisplay.textContent = formatTime(remainingSeconds);

        // Visual warnings
        timerDisplay.classList.remove('warning', 'danger');
        if (remainingSeconds <= 30) {
            timerDisplay.classList.add('danger');
        } else if (remainingSeconds <= 60) {
            timerDisplay.classList.add('warning');
        }
    }

    function tick() {
        if (remainingSeconds <= 0) {
            clearInterval(timerInterval);
            timerDisplay.textContent = '0:00';
            timerDisplay.classList.add('danger');
            // Don't auto-submit — let the learner finish their thought
            finishBtn.textContent = "Time's Up — Submit";
            finishBtn.classList.add('btn-warning');
            return;
        }
        remainingSeconds--;
        updateTimerDisplay();
    }

    function startTimer() {
        updateTimerDisplay();
        timerInterval = setInterval(tick, 1000);
    }

    // --- Event Listeners ---
    writingInput.addEventListener('input', () => {
        updateWordCount();
        saveDraft();
    });

    // Ctrl+Enter to submit
    writingInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            writingForm.submit();
        }
    });

    // Clear draft on form submit
    if (writingForm) {
        writingForm.addEventListener('submit', () => {
            clearDraft();
        });
    }

    // --- Initialize ---
    restoreDraft();
    updateWordCount();
    startTimer();
    writingInput.focus();
})();
