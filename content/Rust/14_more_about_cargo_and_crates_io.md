---
share: true
---
# More About Cargo and Crates.io

The notes reflect the topics in
[Chapter 14](https://doc.rust-lang.org/book/ch14-00-more-about-cargo.html)

This chapter covers some nice features about Cargo and crates.io. It's worth a
chapter here because you may be shocked how hard it is for other languages to do
the same thing.

## Customizing Builds with Release Profiles

`cargo` has its own `rustc -O` support. You spefify these in your `Cargo.toml`

```toml
[profile.dev]
opt-level = 0

[profile.release]
opt-level = 3
```

and you get to use them in `cargo build` and `cargo build --release`.

This is similar to
[Bazel configuration's](https://docs.bazel.build/versions/main/guide.html#--config)
`--config`.

## Publishing a Crate to Crates.io

### Document and Doctest

We have [Sphinx](https://www.sphinx-doc.org/en/master/) in Python land,
[JavaDoc :face_vomiting:](https://docs.oracle.com/javase/8/docs/technotes/tools/windows/javadoc.html)
in, of course, Java land. In C++ we usually use
[Doxygen](https://www.doxygen.nl/).

> I like the auto generate doc. It looks modern and easy to navigate.
>
> -- <cite>Absolutely noone</cite>

Ok, maybe [swagger](https://swagger.io/) is cool (for OpenAPI). The style of
[Read the Docs](https://readthedocs.org/) isn't so bad. And I'm sure about the
JavaScript / TypeScript / NodeJS / Deno kids has more fancy stuff, maybe.

Anyway, we have a nice-looking `rustdoc` tool here.

````rust
//! # My Crate // (1)
//!
//! `my_crate` is a collection of utilities to make performing certain
//! calculations more convenient.
/// Adds one to the number given. // (2)
///
/// # Examples
///
/// ```
/// let arg = 5;
/// let answer = my_crate::add_one(arg);
///
/// assert_eq!(6, answer);
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}
````

1. `//!` would make a crate document.
2. Note it's 3 `/`, not 2 here, so this would show up in doc page.

To see the HTML doc, run `cargo doc [--open]`. Note this is also a good way to
see the doc of your included dependencies.

The example there is included as doc test in `cargo test`.

### pub use

To expose your internal stuff buried in module, use `pub use` in your main file.
This is like `import foo.bar as bar` in `__init__.py` in Python.

## Publishing a crate

`cargo publish`.

Doing a comparison here is painful. Eh, I don't want to write about it. In
Python you got `setup.py`, `sdist`, `bdist`, `twine`, so complicated (or you can
use `poetry publish`). I used to use [nwb](https://github.com/insin/nwb) for
publish my React.JS components, not sure what's the current state.

## Cargo Workspace

You can have a workspace containing multiple packages. This is `Cargo.toml`

```toml
[workspace]

members = [
    "adder",
]
```

Then inside it you do your normal `cargo new adder`. The output directory would
be shared and common dependencies on the same version.

## More cargo

`cargo install` is same as `npm install -g`. Like `git`, it recognize binaries
like `cargo-something`.
