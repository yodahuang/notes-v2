---
share: true
---
# Using Structs to Structure Related Data

The notes reflect the topics in
[Chapter 5](https://doc.rust-lang.org/book/ch05-00-structs.html)

TL;DR: Struct is just `class` in C++, but blends in some Python `dataclass`
goodies and some ECMAScript features.

## A normal struct

```rust
#[derive(Debug)] // (1)
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle { // (3)
    fn area(&self) -> u32 {
        self.width * self.height
    }
    fn square(size: u32) -> Rectangle { // (2)
        Rectangle {
            width: size,
            height: size,
        }
    }}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    let _rect2 = Rectangle::square(3);

    println!("rect1 is {:?}", rect1);
    println!("rect1 is {rect1:?}") // f-string style now work out of box.
    dbg!(&rect1); // (4)

    println!(
        "The area of the rectangle is {} square pixels.",
        rect1.area()
    );
}
```

1. This implements the `Debug` trait. Essentially this offers funcionality
   similar to `__repr__` in Python `dataclass`. Later in that `println!` we see
   `{:?}`. This means we want to use an output format called `Debug`.

2. A static function. But no special `static` keyword, or `@staticmethod`. Here
   is is called associated functions.

3. Method are all put inside this `impl` block. How come Rust doesn't need to
   seperate header and implementation? This is a quesiton I'm not sure (Stack
   Overflow / Duckduckgo is not helping here). Java doesn't need that too. My
   guess is the generated binary contains that signature information. Thus in
   linking time, the linker can just extract the "header" information from the
   object. Also there's no preprocessor (is macro in Rust doing that?) so stuff
   is simpler.

4. A sweet macro that pretty prints the object, together with the line of code,
   line number, etc. Kinda like [icecream](https://github.com/gruns/icecream) in
   Python.

## JavaScript-like syntax

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn build_user(email: String, username: String) -> User {
    User {
        email, // (1)
        username,
        active: true,
        sign_in_count: 1,
    }
}

fn main() {
    let user1 = build_user(
        String::from("someone@example.com"),
        String::from("someusername123"),
    );

    let user2 = User {
        email: String::from("another@example.com"),
        ..user1 // (2)
    };
}
```

1. Field init shorthand, which is like
   [shorthand property name in ES2015](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer#property_definitions)
2. Struct update syntax, which is like
   [spread in object literals in ES2018](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax#spread_in_object_literals).
   Note that the order is different though. In Rust, important object comes
   first, while in ECMAScript it comes last.

## Some special ones

```rust
fn main() {
    // Tuple struct. Just give tuples distint types.
    struct Color(i32, i32, i32);
    struct Point(i32, i32, i32);

    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);

    // Unit-like struct, mostly used in traits.
    struct AlwaysEqual;
    let subject = AlwaysEqual;
}

```
