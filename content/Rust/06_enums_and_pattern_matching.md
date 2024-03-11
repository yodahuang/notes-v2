---
share: true
---
# Enums and Pattern Matching

The notes reflect the topics in
[Chapter 6](https://doc.rust-lang.org/book/ch06-00-enums.html)

Enum is not the `enum` in C++ or `Enum` in Python. If it has to be state in the
term of C++, it's like a combination of `std::variant` + class enum +
`std::tuple`. To state in other word, enums contains variants of types that can
holds other data.

```rust
#[derive(Debug)]
enum Message { // (1)
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        println!("The message is {:?}", self)
    }
}

fn main() {
    let m = Message::Write(String::from("hello"));
    m.call();
}
```

1. This is similar to
    ```rust
    struct QuitMessage; // unit struct
    struct MoveMessage {
        x: i32,
        y: i32,
    }
    struct WriteMessage(String); // tuple struct
    struct ChangeColorMessage(i32, i32, i32); // tuple struct
    ```
    but the data stored can not be retrieved trivially, like:
    ```rust
    let m = Message::Move(6, 9);
    println!(m.x);
    ```
    would not compile. This is because the type of `m` is `Message`, not a
    custom type. To get the data out you'll need some match.

## Another example of pattern matching

```rust
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // --snip--
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
        // (1)
    }
}

fn main() {
    value_in_cents(Coin::Quarter(UsState::Alaska));

    // Showcase the usage of if let
    let coin = Coin::Penny;
    let mut count = 0;
    if let Coin::Quarter(state) = coin {
        println!("State quarter from {:?}!", state);
    } else {
        count += 1;
    }
}


```

1.  matches are exhaustive. You can use `others => foo(others)` or `_ => 42` to
    handle the default cases.

## The Option enum

The `Option` in stdlib looks like this:

```rust
enum Option<T> {
    None,
    Some(T),
}

// Use like this
fn main() {
    let some_number = Some(5);
    let some_string = Some("a string");

    let absent_number: Option<i32> = None;
}
}
```

## Rant time (about C++)

Enjoy my freshed made meme:

![State of C++ pattern matching](https://i.imgflip.com/5zcq15.jpg)

See also the great blog post:
[std::visit is everything wrong with modern C++](https://bitbashing.io/std-visit.html)
