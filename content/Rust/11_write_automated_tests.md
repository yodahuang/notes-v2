---
share: true
---
# Writing Automated Tests

The notes reflect the topics in
[Chapter 11](https://doc.rust-lang.org/book/ch11-00-testing.html)

## Unit test

Rust / Cargo provides built-in support for unit test and integration test. The
unit test lib is just like `unit-test` in Python with nice threading support
just like `pytest`.

```rust
// You can have the normal functions above
use super::*; // (4)

#[cfg(test)] // (1)
mod tests {
    #[test] // (2)
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
    #[test]
    fn it_works_too() -> Result<(), String> { // (3)
        if 2 + 2 == 4 {
            Ok(())
        } else {
            Err(String::from("two plus two does not equal four"))
        }
    }
}

```

1. The `#[cfg(test)]` annotation on the tests module tells Rust to compile and
   run the test code only when you run cargo test, not when you run cargo build.
   The attribute cfg stands for _configuration_ and tells Rust that the
   following item should only be included given a certain configuration option.
   In this case, the configuration option is test, which is provided by Rust for
   compiling and running tests.
2. This tells `cargo test` for what function shall be tested.

3. Yeah we can use `Result` here.

4. In this example, it does not do anything. But you get the idea. Because the
   tests module is an inner module, we need to bring the code under test in the
   outer module into the scope of the inner module. We use a glob here so
   anything we define in the outer module is available to this tests module.

## Integration Tests

Put the test fies in `test` directory.
