---
share: true
---
# An I/O Project: Building a Command Line Program

This chapter is basically a guided homework. This note would document the
progress of me folling the
[guide](https://doc.rust-lang.org/book/ch12-00-an-io-project.html).

## Accepting Command Line Arguments

```rust
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    dbg!(args); // (1)

    let query = &args[1]; // (2)
    let file_path = &args[2];
    println!("Searching for {}", query);
    println!("In file {}", file_path);
}
```

1. Debug trait is defined for vector, while the normal display one is not. Make
   sense.
2. What if we change this to `args[1]`? There will be an error "Cannot move out
   of index of Vec<String>". This is there because a borrowed value cannot be
   move out, and the index operation does return a reference, so it's borrowed.

    It make sense because it's strange that some value in a vector is moved out
    while others are not.

## Reading the file + refactor

See [This part in Chapter 7] for the organization of the code: a `main.rs` and a
`lib.rs`, each with the name of the package, and the `main.rs` use `lib.rs` just
like any other normal client.

=== "main.rs"

    ```rust
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::process;

    use minigrep::Config; // (1)

    fn run(config: Config) -> Result<(), Box<dyn Error>> { // (2)
        println!("Searching for {}", config.query);
        println!("In file {}", config.file_path);
        let contents = fs::read_to_string(config.file_path)?;
        println!("With text:\n{contents}");
        Ok(())
    }

    fn main() {
        let args: Vec<String> = env::args().collect();

        let config = Config::build(&args).unwrap_or_else(|err| { // (3)
            println!("Problem parsing arguments: {err}");
            process::exit(1);
        });

        if let Err(e) = run(config) { // (4)
            println!("Application error {e}");
            process::exit(1);
        }
    }
    ```

    1. Note how stuff in `lib.rs` is imported to the namespace via `minigrep::`, not
    `lib::`. That's because we got a crate for the binary and a crate for the
    lib. Here we're just refering to the lib crate.

    2. `()` means unit type, which is `void` in C++. `Box<dyn Error>` means any
    Error that implements `Error` traint. More on that later. Note with this
    here, we cannot return stuff like `Err("Hey hey")`, as the `str` type is used
    as `E` in the generic data type in `Result<T, E>`, and it dose not implement
    the `Error` trait. We don't enforce the `E` type to implement `Error`.

    3. `unwrap_or_else` hasn't been covered yet. This is equivalent to this in C++:
        ```c++
        const auto status_result = some_operation();
        if (!status_result.ok()) {
            handle_it();
        }
        ```

    4. No need to `unwrap_or_else` here since we don't need the return value.

=== "lib.rs"

    ```rust
    pub struct Config {
        pub query: String,
        pub file_path: String,
    }

    impl Config {
        pub fn build(args: &[String]) -> Result<Config, &'static str> { // (1)
            if args.len() < 3 {
                return Err("Not enough arguments");
            }
            Ok(Config {
                query: args[1].clone(), // (2)
                file_path: args[2].clone(),
            })
        }
    }
    ```

    1. string literals have lifetime of `'static'`. It's a const string slice, represented as `&str`. As far as I know there's only const static string slice and the ones pointing inside `String`.
    2. use `clone` here so the lifetime can be detached from the input args, and now `Config` is the owner of the strings.

## Add functionality and test

```rust
pub fn search<'a>(query: &str, content: &'a str) -> Vec<&'a str> { // (1)
    let mut result = Vec::new();
    for line in content.lines() {
        if line.contains(query) {
            result.push(line); // (2)
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_result() {
        let query = "duct"; // (3)
        let contents = "\
Rust:
safe, fast, productive.
Pick three.";

        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }
}
```

1. Don't forget the `<'a>` after the function name.
2. No `&` before the line here since it's already a reference.
3. Th `\` next line prevents a new line from inserted at the first line.

## Rest (env var, case insensitive, stderr)

Pretty basic. Not covered here.
