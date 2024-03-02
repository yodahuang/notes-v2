---
share: true
---
# Patterns and Matching

The notes reflect the topics in
[Chapter 18](https://doc.rust-lang.org/book/ch18-00-patterns.html)

## All the places patterns can be used.

```rust
// Simple match
match x {
    None => None,
    Some(i) => Some(i + 1),
}

// if let
let favorite_color: Option<&str> = None;
let is_tuesday = false;
let age: Result<u8, _> = "34".parse();

if let Some(color) = favorite_color {
    println!("Using your favorite color, {color}, as the background");
} else if is_tuesday {
    println!("Tuesday is green day!");
} else if let Ok(age) = age {
    if age > 30 {
        println!("Using purple as the background color");
    } else {
        println!("Using orange as the background color");
    }
} else {
    println!("Using blue as the background color");
}

// while let
let mut stack = Vec::new();

stack.push(1);
stack.push(2);
stack.push(3);

while let Some(top) = stack.pop() {
    println!("{}", top);
}

// for loops
let v = vec!['a', 'b', 'c'];

for (index, value) in v.iter().enumerate() {
    println!("{} is at index {}", value, index);
}

// let statement
let (x, y, z) = (1, 2, 3);

// function parameters
fn print_coordinates(&(x, y): &(i32, i32)) {
    println!("Current location: ({}, {})", x, y);
}

fn main() {
    let point = (3, 5);
    print_coordinates(&point);
}
```

## Refutability

Just an idea so you know what it means in error message.

> Patterns come in two forms: refutable and irrefutable. Patterns that will
> match for any possible value passed are irrefutable. An example would be `x`
> in the statement `let x = 5`; because `x` matches anything and therefore
> cannot fail to match. Patterns that can fail to match for some possible value
> are refutable. An example would be `Some(x)` in the expression
> `if let Some(x) = a_value `because if the value in the `a_value` variable is
> `None `rather than `Some`, the `Some(x)` pattern will not match.

## Pattern Syntax

So this is mostly reference.

```rust
// Interesting syntax
match x {
    1 | 2 => println!("one or two"),
    3..7 => println!("three to 7"),
    _ => println!("anything"),
}

// Struct
let p = Point { x: 0, y: 7 };

let Point { x: a, y: b } = p;
// we can also use the same variable name
let Point { x, y } = p;
// We can even only destruct part of the variable
match p {
    Point { x, y: 0 } => println!("On the x axis at {}", x),
    Point { x: 0, y } => println!("On the y axis at {}", y),
    Point { x, y } => println!("On neither axis: ({}, {})", x, y),
}


// Enum
enum Message {
    Quit,
    Move { x: i32, y: i32 }, // struct-like
    Write(String), // tuple-like
    ChangeColor(i32, i32, i32), // Also tuple-like
    ChangeColorPro(Color),
}

enum Color {
    Rgb(i32, i32, i32),,
    Hsv(i32, i32, i32),
}


fn main() {
    let msg = Message::ChangeColor(0, 160, 255);

    match msg {
        Message::Quit => {
            println!("The Quit variant has no data to destructure.")
        }
        Message::Move { x, y } => {
            println!(
                "Move in the x direction {} and in the y direction {}",
                x, y
            );
        }
        Message::Write(text) => println!("Text message: {}", text),
        Message::ChangeColor(r, g, b) => println!(
            "Change the color to red {}, green {}, and blue {}",
            r, g, b
        ),
        Message::ChangeColorPro(Color::Rgb(r, g, b)) => println!(
            "Change the color to red {}, green {}, and blue {}",
            r, g, b
        ),
    }
}

```

`_` can be used as a wild pattern to ignore whatever value is there.

We can tell the compiler to ignore unused value by prefixing with `_`. Something
like `let _x = 5`.

We can even ignore using `..` like this:

```rust
let numbers = (2, 4, 8, 16, 32);

match numbers {
    (first, .., last) => {
        println!("Some numbers: {first}, {last}");
    }
}
```

It even has match guards :exploding_head:.

```rust
    let num = Some(4);

    match num {
        Some(x) if x % 2 == 0 => println!("The number {} is even", x),
        Some(x) => println!("The number {} is odd", x),
        None => (),
    }
```

In C++ land I guess we can have custom trait with some `std::enable_if_t` and
`std::is_same_of`.

We can also bind whatever is matched in a variable by using `@`.

```rust
fn main() {
    enum Message {
        Hello { id: i32 },
    }

    let msg = Message::Hello { id: 5 };

    match msg {
        Message::Hello {
            id: id_variable @ 3..=7,
        } => println!("Found an id in range: {}", id_variable),
        Message::Hello { id: 10..=12 } => {
            println!("Found an id in another range")
        }
        Message::Hello { id } => println!("Found some other id: {}", id),
    }
}
```

Note how in the first arm we are able to both match and store the value into
`id_variable`.
