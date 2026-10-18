/**
 * RAG Proof Box & Interactive Highlight Component
 * 
 * Provides an instant hover popover for AI-highlighted words with authoritative
 * proof, source citation, and direct reference links (e.g. Merriam-Webster).
 * Ensures immediate dismissal upon unhovering ("when unhover it will gone immediately").
 */

(function () {
    'use strict';

    function initRagProofBox() {
        const highlightWords = document.querySelectorAll('.ai-highlight-word');
        if (!highlightWords.length) return;

        // Ensure proof box container exists in DOM
        let popover = document.getElementById('rag-proof-box');
        if (!popover) {
            popover = document.createElement('div');
            popover.id = 'rag-proof-box';
            popover.className = 'rag-proof-popover';
            popover.setAttribute('role', 'tooltip');
            popover.setAttribute('aria-hidden', 'true');
            popover.style.display = 'none';

            popover.innerHTML = `
                <div class="rag-proof-arrow"></div>
                <div class="rag-proof-inner">
                    <div class="rag-proof-header">
                        <div class="rag-proof-badges">
                            <span class="badge-source" id="rag-box-source">🏛️ Merriam-Webster</span>
                            <span class="badge-category" id="rag-box-category">Grammar</span>
                        </div>
                        <div class="rag-proof-rule-title" id="rag-box-rule">Indefinite Articles</div>
                    </div>
                    <div class="rag-proof-diff">
                        <span class="diff-original" id="rag-box-original">a</span>
                        <span class="diff-arrow">➔</span>
                        <span class="diff-correction" id="rag-box-correction">an</span>
                    </div>
                    <div class="rag-proof-explanation" id="rag-box-explanation">
                        Explanation
                    </div>
                    <div class="rag-proof-citation">
                        <div class="citation-label">Authoritative Proof:</div>
                        <blockquote class="citation-text" id="rag-box-proof">
                            Proof citation
                        </blockquote>
                    </div>
                    <div class="rag-proof-footer">
                        <a id="rag-box-link" href="#" target="_blank" rel="noopener noreferrer" class="rag-proof-btn">
                            Official Reference ↗
                        </a>
                    </div>
                </div>
            `;
            document.body.appendChild(popover);
        }

        const sourceEl = document.getElementById('rag-box-source');
        const categoryEl = document.getElementById('rag-box-category');
        const ruleEl = document.getElementById('rag-box-rule');
        const originalEl = document.getElementById('rag-box-original');
        const correctionEl = document.getElementById('rag-box-correction');
        const explanationEl = document.getElementById('rag-box-explanation');
        const proofEl = document.getElementById('rag-box-proof');
        const linkEl = document.getElementById('rag-box-link');

        let activeTarget = null;
        let isOverPopover = false;

        function showPopover(target) {
            activeTarget = target;

            // Extract data attributes
            const original = target.dataset.original || target.textContent.trim();
            const correction = target.dataset.correction || '(correction)';
            const category = target.dataset.category || 'Grammar';
            const ruleName = target.dataset.ruleName || 'Standard English Rule';
            const sourceName = target.dataset.sourceName || 'Merriam-Webster';
            const sourceUrl = target.dataset.sourceUrl || `https://www.merriam-webster.com/dictionary/${encodeURIComponent(original)}`;
            const proof = target.dataset.proof || target.dataset.explanation || 'Authoritative usage guidance.';
            const explanation = target.dataset.explanation || '';
            const mistakeIndex = target.dataset.index;

            // Populate popover
            sourceEl.textContent = `🏛️ ${sourceName}`;
            categoryEl.textContent = category;
            ruleEl.textContent = ruleName;
            originalEl.textContent = original;
            correctionEl.textContent = correction;
            explanationEl.textContent = explanation;
            proofEl.textContent = `“${proof}”`;
            linkEl.href = sourceUrl;
            linkEl.textContent = `View on ${sourceName} ↗`;

            // Make visible to measure dimensions
            popover.style.display = 'block';
            popover.style.visibility = 'hidden';
            popover.setAttribute('aria-hidden', 'false');

            // Position popover
            const rect = target.getBoundingClientRect();
            const popoverRect = popover.getBoundingClientRect();
            const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
            const scrollY = window.pageYOffset || document.documentElement.scrollTop;

            const margin = 10;
            let top = rect.top + scrollY - popoverRect.height - margin;
            let placement = 'top';

            // If not enough room on top, position below
            if (rect.top - popoverRect.height - margin < 10) {
                top = rect.bottom + scrollY + margin;
                placement = 'bottom';
            }

            // Center horizontally relative to target
            let left = rect.left + scrollX + (rect.width / 2) - (popoverRect.width / 2);

            // Clamp inside viewport
            const minLeft = scrollX + 12;
            const maxLeft = scrollX + document.documentElement.clientWidth - popoverRect.width - 12;
            left = Math.max(minLeft, Math.min(left, maxLeft));

            popover.style.top = `${top}px`;
            popover.style.left = `${left}px`;
            popover.className = `rag-proof-popover placement-${placement}`;
            popover.style.visibility = 'visible';

            // Sync with sidebar card if available
            if (mistakeIndex !== undefined) {
                const mistakeCards = document.querySelectorAll('.mistake-card');
                mistakeCards.forEach((card, idx) => {
                    if (String(idx) === String(mistakeIndex)) {
                        card.classList.add('highlighted-by-rag');
                        // If pager is used and hidden, reveal it
                        if (card.style.display === 'none') {
                            const currentLabel = document.getElementById('mistake-current');
                            const prevBtn = document.getElementById('mistake-prev');
                            const nextBtn = document.getElementById('mistake-next');
                            mistakeCards.forEach((c, i) => {
                                c.style.display = i === idx ? '' : 'none';
                            });
                            if (currentLabel) currentLabel.textContent = idx + 1;
                            if (prevBtn) prevBtn.disabled = idx === 0;
                            if (nextBtn) nextBtn.disabled = idx === mistakeCards.length - 1;
                        }
                    } else {
                        card.classList.remove('highlighted-by-rag');
                    }
                });
            }
        }

        // Hide popover immediately when unhovering
        function hidePopoverImmediately() {
            activeTarget = null;
            popover.style.display = 'none';
            popover.style.visibility = 'hidden';
            popover.setAttribute('aria-hidden', 'true');

            // Remove highlight from cards
            document.querySelectorAll('.mistake-card.highlighted-by-rag').forEach(card => {
                card.classList.remove('highlighted-by-rag');
            });
        }

        // Event listeners for highlighted words
        highlightWords.forEach(word => {
            word.addEventListener('mouseenter', () => {
                showPopover(word);
            });

            word.addEventListener('mouseleave', (e) => {
                // If moving directly into the popover, don't close yet so user can click the reference link
                const toElement = e.relatedTarget;
                if (toElement && (toElement === popover || popover.contains(toElement))) {
                    return;
                }
                // Unhovered: gone immediately!
                hidePopoverImmediately();
            });

            // Keyboard accessibility
            word.addEventListener('focus', () => showPopover(word));
            word.addEventListener('blur', hidePopoverImmediately);
        });

        // Allow interacting with the popover (clicking link)
        popover.addEventListener('mouseenter', () => {
            isOverPopover = true;
        });

        popover.addEventListener('mouseleave', (e) => {
            isOverPopover = false;
            // If mouse moved back onto the active target, keep it
            const toElement = e.relatedTarget;
            if (toElement && activeTarget && (toElement === activeTarget || activeTarget.contains(toElement))) {
                return;
            }
            // Otherwise gone immediately!
            hidePopoverImmediately();
        });

        // Dismiss on Escape key or window scroll
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') hidePopoverImmediately();
        });

        window.addEventListener('scroll', () => {
            if (popover.style.display !== 'none' && !isOverPopover) {
                hidePopoverImmediately();
            }
        }, { passive: true });
    }

    // Auto-initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initRagProofBox);
    } else {
        initRagProofBox();
    }
})();
