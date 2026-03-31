(function () {
    const storedTheme = localStorage.getItem("talentlens-theme");
    const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = storedTheme || (preferredDark ? "dark" : "light");

    document.documentElement.setAttribute("data-theme", theme);

    function syncButtons() {
        document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
            const isDark = document.documentElement.getAttribute("data-theme") === "dark";
            button.textContent = isDark ? "Light Mode" : "Dark Mode";
            button.setAttribute("aria-pressed", String(isDark));
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        syncButtons();
        document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
            button.addEventListener("click", () => {
                const current = document.documentElement.getAttribute("data-theme") || "light";
                const next = current === "dark" ? "light" : "dark";
                document.documentElement.setAttribute("data-theme", next);
                localStorage.setItem("talentlens-theme", next);
                syncButtons();
            });
        });
    });
})();
