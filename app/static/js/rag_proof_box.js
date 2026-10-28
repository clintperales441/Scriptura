/**
 * RAG Proof Box & Interactive Highlight Component
 * 
 * Provides an instant hover popover for AI-highlighted sentences/words with
 * authoritative proof, part-of-speech breakdown chips, source citation, and
 * direct reference links (Merriam-Webster), synced with the detailed card pager.
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
                    <div class="rag-proof-pos-strip" id="rag-box-pos-strip" style="display: none;"></div>
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
        const posStripEl = document.getElementById('rag-box-pos-strip');
        const proofEl = document.getElementById('rag-box-proof');
        const linkEl = document.getElementById('rag-box-link');

        let activeTarget = null;
        let isOverPopover = false;

        function renderPosChips(rawBreakdown) {
            if (!posStripEl) return;
            posStripEl.innerHTML = '';
            if (!rawBreakdown) {
                posStripEl.style.display = 'none';
                return;
            }
            try {
                const items = JSON.parse(rawBreakdown);
                if (!Array.isArray(items) || !items.length) {
                    posStripEl.style.display = 'none';
                    return;
                }
                items.slice(0, 6).forEach((entry) => {
                    const chip = document.createElement('span');
                    chip.className = 'rag-pos-chip';
                    const posLabel = entry.pos || 'Grammar';
                    const tokenLabel = entry.sugg && entry.sugg !== '(removed)'
                        ? entry.sugg
                        : `−${entry.orig || ''}`;
                    chip.textContent = `${posLabel}: ${tokenLabel}`;
                    posStripEl.appendChild(chip);
                });
                if (items.length > 6) {
                    const more = document.createElement('span');
                    more.className = 'rag-pos-chip rag-pos-chip-more';
                    more.textContent = `+${items.length - 6} more`;
                    posStripEl.appendChild(more);
                }
                posStripEl.style.display = 'flex';
            } catch (_err) {
                posStripEl.style.display = 'none';
            }
        }

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
            renderPosChips(target.dataset.breakdown);
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

            // Sync with sidebar card pager if available
            if (mistakeIndex !== undefined) {
                if (typeof window.selectMistakeCard === 'function') {
                    window.selectMistakeCard(mistakeIndex);
                }
                const mistakeCards = document.querySelectorAll('.mistake-card');
                mistakeCards.forEach((card, idx) => {
                    if (String(idx) === String(mistakeIndex)) {
                        card.classList.add('highlighted-by-rag');
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

            word.addEventListener('click', () => {
                if (word.dataset.index !== undefined && typeof window.selectMistakeCard === 'function') {
                    window.selectMistakeCard(word.dataset.index);
                }
            });

            word.addEventListener('mouseleave', (e) => {
                const toElement = e.relatedTarget;
                if (toElement && (toElement === popover || popover.contains(toElement))) {
                    return;
                }
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
            const toElement = e.relatedTarget;
            if (toElement && activeTarget && (toElement === activeTarget || activeTarget.contains(toElement))) {
                return;
            }
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
