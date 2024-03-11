---
share: true
---
# Managing Growing Projects with Packages, Crates, and Modules

The notes reflect the topics in
[Chapter 7](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html)

## Official definition and my definition

> -   Packages: A Cargo feature that lets you build, test, and share crates
> -   Crates: A tree of modules that produces a library or executable
> -   Modules and `use`: Let you control the organization, scope, and privacy of
>     paths
> -   Paths: A way of naming an item, such as a struct, function, or module

-   Packages: The thing you install via `cargo`. Not related to Rust compiler,
    just a something for the package manager. It can contain a library or
    multiple binaries. Here the library / binary are what we call crates. It's
    similar to a Python package that you create via `setup.py` (which can have
    multiple `entry_points`)
-   Crates: The library / binary itself. It's the compilation unit (unlike C++,
    where we compile each file individually and then link them together). Think
    of it as a [Bazel](https://bazel.build) build target.
-   Modules: Python module system blend with C++ namespace. Like Python, it
    organize the structure of the "namespace" by file path. That means function
    `baz` in `foo/bar` can be used as `use crate::foo::bar` and then `bar::baz`.
    More on that later. Like namespace, you write things like
    ```rust
    mod foo {
        fn bar() {}
    }
    ```
-   Path: That `crate::foo::bar` thing.

## Example time

```rust
mod front_of_house {
    pub mod hosting { // (1)
        pub fn add_to_waitlist() {}
    }
}

mod back_of_house {
    pub struct Breakfast {
        pub toast: String, // (2)
        seasonal_fruit: String,
    }

    pub enum Appetizer { // (3)
        Soup,
        Salad,
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
}

pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();

    // Relative path
    front_of_house::hosting::add_to_waitlist();

    // Order a breakfast in the summer with Rye toast
    let mut meal = back_of_house::Breakfast::summer("Rye");
    // Change our mind about what bread we'd like
    meal.toast = String::from("Wheat");
    println!("I'd like {} toast please", meal.toast);

    // The next line won't compile if we uncomment it; we're not allowed
    // to see or modify the seasonal fruit that comes with the meal
    // meal.seasonal_fruit = String::from("blueberries");
}
```

1. ### The keyword `pub` and everything
    `pub` basically means "make this thing publically accessible outside this
    module". It's like
    [`module.exports`](https://nodejs.org/api/modules.html#moduleexports) in
    Node.js, but it also applies to struct variable / method. So there's no
    `public` / `private` / `protected` in Rust. Instead, all the things are
    "public" within a module. Well, you can argue that in C++, each class has
    its own namespace, so all the things are also "public" in that class
    namespace. Outside of that module though, all things are default private
    unless being annotated with `pub`.
2. This `toast` is marked visible outside of module (`back_of_house`), while
   `seasonal_fruit` is not.

3. For enum, `pub` only need to be declared once.

`use` is like `import` in Python. By `like` I mean not only do they both import
some other lib, but they are similar in the term of handling file path and
importing submodules.

```rust
use std::io::{self, Write};
use std::io::Result as IoResult;

mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub use self::front_of_house::hosting; // (1)

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
}

```

1. This re-export this to the user of this crate.

Another example with the content of the module spread in different files

=== "src/lib.rs"

    ```rust
    mod front_of_house; // (1)

    pub use crate::front_of_house::hosting;

    pub fn eat_at_restaurant() {
        hosting::add_to_waitlist();
        hosting::add_to_waitlist();
        hosting::add_to_waitlist();
    }
    ```

    1. Using a semicolumn teels Rust to load the contents of the module from another file with the same name as the module. In this case, that's just `front_of_house.rs`

=== "src/front_of_house.rs"

    ```rust
    pub mod hosting
    ```

=== "src/front_of_house/hosting.rs"

    ```rust
    pub fn add_to_waitlist() {}
    ```

## A nice little table

I'm not familiar with C++ 20 modules. The C++ here is referring to the old-style
`import foo/bar.h`.

| Language                                                         | Rust                                                               | C++                                                                                                   | Python                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| What's exported in a package                                     | Whatever you put a `pub` on.                                       | All the things in the header file.                                                                    | All the things.                                                               |
| Class member public / private                                    | All public in module, default private outside module.              | Default public / private depends on whether you use `class` or `struct`. It's set at the class level. | You've got no choice. All public. Users are suppose to respect `_foo` though. |
| Re-export imported libs                                          | You can choose to reexport them, but by default it's not exported. | You've got no choice. Export all the way.                                                             | You've got no choice. Export all the way.                                     |
| Can you ctrl-f on a symbol to figure out where does it come from | :white_check_mark:                                                 | :x:                                                                                                   | :white_check_mark:                                                            |

## use vs mod

All the crates added in as dependency by Cargo is already in scope. So we don't
need to `mod` them first. But for the module / submodues we have ourselves, we
still need to use `mod` to tell Rust "search here" first.
