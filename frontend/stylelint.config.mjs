/** @type {import("stylelint").Config} */
export default {
  extends: ["stylelint-config-standard"],
  plugins: ["stylelint-order", "stylelint-declaration-strict-value"],
  rules: {
    "order/properties-alphabetical-order": true,
    "scale-unlimited/declaration-strict-value": [
      ["/color$/", "fill", "stroke", "background", "z-index", "font-family"],
      { ignoreFunctions: false, ignoreValues: ["currentColor", "inherit", "transparent", "none", "initial", "unset"] },
    ],
    // Tailwind v4 expands these directives and utility lists before serving CSS.
    "at-rule-no-unknown": [true, {ignoreAtRules: ["theme", "custom-variant", "apply", "plugin", "source"]}],
    "at-rule-prelude-no-invalid": [true, {ignoreAtRules: ["apply"]}],
    // Vite resolves package imports; they are not literal stylesheet URLs.
    "import-notation": "string",
    // Compact keyframes keep their single-line form.
    "declaration-block-single-line-max-declarations": null,
  },
};
