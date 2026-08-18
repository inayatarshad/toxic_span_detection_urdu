/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ivory: "#FBF9F4",
        cream: "#F4EEE2",
        sand: {
          DEFAULT: "#E5D9C6",
          deep: "#D6C7AE",
        },
        forest: {
          DEFAULT: "#26342B",
          deep: "#18221C",
          mid: "#3A4E41",
          soft: "#5C7264",
        },
        merlot: {
          DEFAULT: "#3F0B0D",
          mid: "#6E1B1F",
          bright: "#8F3438",
          wash: "#F0DFDC",
        },
      },
      fontFamily: {
        serif: ['Fraunces', 'Georgia', 'Cambria', 'serif'],
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        urdu: ['"Noto Nastaliq Urdu"', '"Jameel Noori Nastaleeq"', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      maxWidth: {
        prose: "68ch",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: 0, transform: "translateY(12px)" },
          to: { opacity: 1, transform: "translateY(0)" },
        },
        "dash-flow": {
          to: { strokeDashoffset: -24 },
        },
        "pulse-node": {
          "0%, 100%": { opacity: 0.25, transform: "scale(1)" },
          "50%": { opacity: 0.55, transform: "scale(1.14)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.5s ease-out both",
        "dash-flow": "dash-flow 1s linear infinite",
        "pulse-node": "pulse-node 2.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
