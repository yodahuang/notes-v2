---
share: true
---
# Error handling

The notes reflect the topics in
[Chapter 9](https://doc.rust-lang.org/book/ch09-00-error-handling.html)

This chapter talks about `panic!` vs `Result<T, F>`, which is essentially
`CHECK` in `gcheck` and `StatusOr<T>` in `Abseil` in C++ land. The chapter
organize the error by _recoverable_ and _unrecoverable_. Other folks
[has argue](https://blog.burntsushi.net/unwrap/#what-about-recoverable-vs-unrecoverable-errors)
that it is better to think of as "is this caused by a bug in the program" or "it
could actually happen", which is also my mindset.

A Hackernews discussion about "Using unwrap() in Rust is okay" can be found
[here](https://news.ycombinator.com/item?id=32385102).

## Unveil bug with `panic!`

It's `CHECK`. Backtrace not saved by default, but can be enabled by envvar
`RUST_BACKTRACE=1`.

```rust
fn main() {
    panic!("crash and burn");
}
```

??? question "Why don't we just always have `RUST_BACKTRACE=1` there?"

    So thers's no performance difference. You can do it if you want. There's discussion
    [at the forum](https://internals.rust-lang.org/t/rust-backtrace-in-production-use/5609).

??? question "Does RAII still holds when it panics?"

    By default, yes, unless you want the binary as small as possible. Then you can add tis to your `Cargo.toml`:
    ```toml
    [profile.release]
    panic = 'abort'
    ```

## Expected error with `Result`

Open a file called `hello.txt`. If it does not exist, create it. This can be
cleaner if using `unwrap_or_else`, but is just here for educational purpose.

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => match error.kind() { // (1)
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {:?}", e),
            },
            other_error => {
                panic!("Problem opening the file: {:?}", other_error);
            }
        },
    };
}
```

1. Ther error type returned is `io::Error`, which is a struct that
   [has method `kind`](https://doc.rust-lang.org/std/io/struct.Error.html#method.kind)
   that helpfully returns an `ErrorKind`. The struct itself does not have public
   fields.

We can simply panic, or having something like `CHECKED_RESULT` by using:

```rust
use std::fs::File;

fn main() {
    let mut greeting_file = File::open("hello.txt").unwrap();
    greeting_file = File::open("hello.txt").expect("hello.txt should be included in this project");
}

```

We can also propagate the error, like you would like `RETURN_OR_ASSIGN` in C++,
or `?` used in
[optional chaining](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-3-7.html)
in TypeScript.

It can work on `Optional<T>` as well.

```rust
fn last_char_of_first_line(text: &str) -> Option<char> {
    text.lines().next()?.chars().last()
}
```

??? note "Let `main` return result"

    If you really want you can let `main` return result:
    ```rust
    use std::error::Error;
    use std::fs::File;

    fn main() -> Result<(), Box<dyn Error>> {
        let greeting_file = File::open("hello.txt")?;

        Ok(())
    }

    ```
