export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function formatSimTime(date) {
  // YYYY-MM-DD HH:MM
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function dist(a, b, c, d) {
  const dx = a - c;
  const dy = b - d;
  return Math.hypot(dx, dy);
}

export function choice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function uniq(arr) {
  return Array.from(new Set(arr));
}
