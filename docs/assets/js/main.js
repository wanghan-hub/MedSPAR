"use strict";

// Progressive enhancement: all research content lives in index.html.
const dialog = document.getElementById("figure-dialog");
const dialogImage = document.getElementById("dialog-image");
const dialogCaption = document.getElementById("dialog-caption");
let figureTrigger = null;

if (dialog && typeof dialog.showModal === "function") {
  document.querySelectorAll("[data-zoom]").forEach((button) => {
    button.hidden = false;
    button.addEventListener("click", () => {
      figureTrigger = button;
      dialogImage.src = button.dataset.zoom;
      dialogImage.alt = button.dataset.caption;
      dialogCaption.textContent = button.dataset.caption;
      dialog.showModal();
    });
  });
  document.getElementById("close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener("close", () => {
    if (figureTrigger) figureTrigger.focus();
  });
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  // Hide unavailable functionality; selecting the code remains possible.
  if (!navigator.clipboard || !window.isSecureContext) return;
  button.hidden = false;
  button.addEventListener("click", async () => {
    const status = document.getElementById("copy-status");
    try {
      await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent.trim());
      button.textContent = "Copied";
      status.textContent = "Preview command copied.";
    } catch {
      status.textContent = "Clipboard is unavailable. Select and copy the command text.";
    }
  });
});

if ("IntersectionObserver" in window) {
  const links = [...document.querySelectorAll("nav a")];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    });
  }, { rootMargin: "-15% 0px -60% 0px", threshold: 0 });
  links.forEach((link) => {
    const section = document.querySelector(link.hash);
    if (section) observer.observe(section);
  });
}
