(function () {
    function autoResize(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.max(textarea.scrollHeight, 140)}px`;
    }

    function wireTextareas(scope) {
        scope.querySelectorAll("textarea").forEach((textarea) => {
            autoResize(textarea);
            textarea.addEventListener("input", () => autoResize(textarea));
        });
    }

    function wireDropzones(scope) {
        scope.querySelectorAll("[data-dropzone]").forEach((dropzone) => {
            const input = dropzone.querySelector('input[type="file"]');
            const label = dropzone.querySelector("[data-file-label]");
            if (!input || !label) {
                return;
            }

            const render = () => {
                const fileNames = Array.from(input.files || []).map((file) => file.name);
                label.textContent = fileNames.length
                    ? fileNames.join(", ")
                    : "Drop a PDF, DOCX, or TXT here, or click to browse";
                dropzone.classList.toggle("has-file", fileNames.length > 0);
            };

            ["dragenter", "dragover"].forEach((eventName) => {
                dropzone.addEventListener(eventName, (event) => {
                    event.preventDefault();
                    dropzone.classList.add("drag-active");
                });
            });

            ["dragleave", "drop"].forEach((eventName) => {
                dropzone.addEventListener(eventName, (event) => {
                    event.preventDefault();
                    if (eventName === "drop") {
                        if (event.dataTransfer?.files?.length) {
                            input.files = event.dataTransfer.files;
                            render();
                        }
                    }
                    dropzone.classList.remove("drag-active");
                });
            });

            input.addEventListener("change", render);
            render();
        });
    }

    function wireHistoryCards(scope) {
        scope.querySelectorAll("[data-history-card]").forEach((card) => {
            card.addEventListener("click", () => {
                card.classList.toggle("expanded");
            });
        });
    }

    function wireModeSwitches(scope) {
        scope.querySelectorAll("[data-mode-switch]").forEach((switcher) => {
            const card = switcher.closest(".candidate-entry") || scope;
            const buttons = switcher.querySelectorAll("[data-mode-target]");
            const panels = card.querySelectorAll("[data-mode-panel]");

            const activate = (targetMode) => {
                buttons.forEach((button) => {
                    button.classList.toggle("is-active", button.dataset.modeTarget === targetMode);
                });
                panels.forEach((panel) => {
                    panel.classList.toggle("is-visible", panel.dataset.modePanel === targetMode);
                });
            };

            buttons.forEach((button) => {
                button.addEventListener("click", () => activate(button.dataset.modeTarget));
            });

            const hasText = Array.from(card.querySelectorAll("textarea")).some(
                (textarea) => textarea.value.trim().length > 0
            );
            activate(hasText ? "paste" : "upload");
        });
    }

    function wireCompareBoard(scope) {
        const compareGrid = scope.querySelector("[data-compare-grid]");
        const emptyState = scope.querySelector("[data-compare-empty]");
        const cards = scope.querySelectorAll("[data-compare-card]");
        if (!compareGrid || !emptyState || !cards.length) {
            return;
        }

        const selected = new Map();

        const render = () => {
            compareGrid.innerHTML = "";
            emptyState.style.display = selected.size ? "none" : "block";

            selected.forEach((candidate) => {
                const article = document.createElement("article");
                article.className = "compare-card";
                article.innerHTML = `
                    <div class="compare-card-head">
                        <div>
                            <strong>${candidate.name}</strong>
                            <p>${candidate.role} | ${candidate.strength}</p>
                        </div>
                        <span class="role-chip recruiter-chip">${candidate.score}%</span>
                    </div>
                    <div class="compare-metrics">
                        <article><span>Job Match</span><strong>${candidate.jobMatch}%</strong></article>
                        <article><span>Coverage</span><strong>${candidate.coverage}%</strong></article>
                    </div>
                    <p><strong>Matched:</strong> ${candidate.matched}</p>
                    <p><strong>Missing:</strong> ${candidate.missing}</p>
                    <p class="muted-line"><strong>Evidence:</strong> ${candidate.highlights}</p>
                `;
                compareGrid.appendChild(article);
            });

            cards.forEach((card) => {
                const toggle = card.querySelector("[data-compare-toggle]");
                const key = card.dataset.name;
                const active = selected.has(key);
                card.classList.toggle("compare-selected", active);
                if (toggle) {
                    toggle.textContent = active ? "Selected" : "Compare";
                    toggle.classList.toggle("is-active", active);
                }
            });
        };

        cards.forEach((card) => {
            const toggle = card.querySelector("[data-compare-toggle]");
            if (!toggle) {
                return;
            }

            toggle.addEventListener("click", () => {
                const key = card.dataset.name;
                if (selected.has(key)) {
                    selected.delete(key);
                    render();
                    return;
                }

                if (selected.size >= 3) {
                    const firstKey = selected.keys().next().value;
                    selected.delete(firstKey);
                }

                selected.set(key, {
                    name: card.dataset.name,
                    role: card.dataset.role,
                    strength: card.dataset.strength,
                    score: card.dataset.score,
                    jobMatch: card.dataset.jobMatch,
                    coverage: card.dataset.coverage,
                    matched: card.dataset.matched,
                    missing: card.dataset.missing,
                    highlights: card.dataset.highlights,
                });
                render();
            });
        });

        render();
    }

    function init(scope) {
        wireTextareas(scope);
        wireDropzones(scope);
        wireHistoryCards(scope);
        wireModeSwitches(scope);
        wireCompareBoard(scope);
    }

    window.TalentLensUI = { init };

    document.addEventListener("DOMContentLoaded", () => {
        init(document);
    });
})();
