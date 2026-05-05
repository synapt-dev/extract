// Negative test: Function() without new — must be caught by check-no-network.mjs
const fn = Function("return 1");
