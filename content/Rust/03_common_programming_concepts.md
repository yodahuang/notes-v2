---
share: true
---
# Common Programming Concepts

The notes reflect the topics in
[Chapter 3](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)

## Variables and Mutability

`const` is the `constexpr` in C++. But, maybe before C++17, where we can use all
kinds of stuff as `constexpr`. C++20 even let us declare
`constexpr vector<int>`! And we can use all the std algorithms on it like
`std::accumulate`... yummy. Rust isn't quite there yet, at least when I write
this note.

```rust
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
```

## Shadowing

```rust
fn main() {
    let x = 5;

    let x = x + 1;

    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {}", x); // 12
    }

    println!("The value of x is: {}", x); // 6
}
```

This is very dope. I always want scope in Python, and it's like a dream come
true.

## Data Types

```rust
fn main() {
    let guess: u32 = "42".parse().expect("Not a number!"); // (1)
    let some_int: i64 = 42_000; // (2)
    let some_float = 2.0; // (3)
    let floored = 2 / 3; // Results in 0 (4)
    let f: bool = false; // with explicit type annotation
    let heart_eyed_cat = '😻';
    let tup: (i32, f64, u8) = (500, 6.4, 1); // type annotation can be ommited
    let arr: [i32; 5] = [1, 2, 3, 4, 5]; // type annotation can be ommitted (5)
}
```

1.  The `u32` must be here, just like C++ behave like this

    ```c++
    const auto guess = parse<uint32_t>("42");
    // or
    const uint32_t guess = parse<>("42"); // The <> can be omitted.
    ```

    I'm not sure what is used in rust in the place of template though.

2.  There's no `int` type in Rust, but `i8`, `i16`, `i32`, etc. This makes
    portability a lot easier (unlike in C++ it's all depending on
    implementation,
    [the standard only requies "minimum ranges"](https://stackoverflow.com/questions/589575/what-does-the-c-standard-state-the-size-of-int-long-type-to-be),
    what's wrong with the committe?).

    It checks for overflow in debug mode, and ignore it in release mode. Clever
    decision and I hope C++ have this too.... All the standard library also
    offer variants to handle overflow, including checking and saturating.

3.  `double` and `float` are just `f64` and `f32` in Rust.

4.  Sadly, `5/2 == 2` in Rust, just like C++ (unlike Python). I can forsee some
    unfortunate numerical error caused by this already.

5.  This is just like `std::array`. The template arguments is in the type
    annotation here. It does support vector-style constructor-like stuff, like
    this:

    ```rust
    let a = [3; 5]; // [3, 3, 3, 3, 3]
    ```

    Rust does have runtime index bound check for array. To avoid the performance
    penalty you may use iterator (introduced later).

## Functions

```rust
fn main() {
    let x = plus_one(5); // (1)

    let y = {
        let x = 3;
        x + 1
    };

}

fn plus_one(x: i32) -> i32 {
    x + 1 // (2)
}

```

1.  Like Python, function can be declared later.
2.  Statement ends with `;` and does not return anything. Expressions doesn't
    have `;` and returns. This is like Matlab. Function has implicit return,
    like Ruby.

It's unclear how rust handle function overloading and templated functions. Also,
is there function object / lambda ?

!!! info "Spoiler! Rust has them, but not exactly"

    There's no function overloading. See this
    [discussion](https://internals.rust-lang.org/t/justification-for-rust-not-supporting-function-overloading-directly/7012)
    for justification. TL;DR is that
    [trait](10_generic_types_traits_and_lifetimes.md#traits) kinda does that, and
    generic does not play along well with function overloading. Wait, what's
    [generic](10_generic_types_traits_and_lifetimes.md#generic-data-types)? That's
    template, without that
    [SFIANE](https://en.cppreference.com/w/cpp/language/sfinae) part. Lambda is
    called
    [closure](13_functional_language_features_iterators_and_closures.md#closures-anonymous-functions-that-capture-their-environment)
    here.

## Control Flow

```rust
fn main() {
    // The normal if / else
    let number = 6;

    if number % 4 == 0 {
        println!("number is divisible by 4");
    } else if number % 3 == 0 {
        println!("number is divisible by 3");
    } else if number % 2 == 0 {
        println!("number is divisible by 2");
    } else {
        println!("number is not divisible by 4, 3, or 2");
    }

    // if is an expression.
    let condition = true;
    let number = if condition { 5 } else { 6 }; // (1)

    // loop and break
    let mut count = 0;
    'counting_up: loop { // (2)
        println!("count = {}", count);
        let mut remaining = 10;

        loop {
            println!("remaining = {}", remaining);
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }

        count += 1;
    }
    println!("End count = {}", count);

    // break can also return value.
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("The result is {}", result);

    // While
    let mut number = 3;

    while number != 0 {
        println!("{}!", number);

        number -= 1;
    }

    // For
    let a = [10, 20, 30, 40, 50];

    for element in a {
        println!("the value is: {}", element);
    }
}
```

1. Like Python `5 if condition else 6`, or C++ `condition? 5 : 6`.
2. Better `goto`
