import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the dataset hero", () => {
  render(<App />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Urdu Toxic/i);
});

test("exposes the primary navigation", () => {
  render(<App />);
  expect(screen.getByRole("navigation", { name: /primary/i })).toBeInTheDocument();
});
