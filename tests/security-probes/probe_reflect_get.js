// Negative test: Reflect.get bypass — must be caught by check-no-network.mjs
const f = Reflect.get(globalThis, "fetch");
