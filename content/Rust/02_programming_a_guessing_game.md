---
share: true
---
# Programming a Guessing Game

The notes reflect the topics in
[Chapter 2](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html)

Just click on the annotaions in the code to see my soupbox.

```rust
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new(); // (1)

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() { // (2) (3) (5)
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {}", guess);

        match guess.cmp(&secret_number) { // (4)
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
```

1.  ### mut vs const

    It's like protobuf. If you want something to be mutable, you need to define
    it. Or, in other ways, `const` everywhere. I like this, as in C++ I wrote
    far more `const auto foo = some_func()` then `auto foo = some_func()`. `mut`
    can also be used in function signature. But let's discuss this in details in
    another note.

2.  ### shadow variables

    No more

    ```c++
    std::string guess_str = some_func();
    uint32_t guess = to_int(std::move(guess_str));
    // If you use guess_str again, undefined behavior.
    ```

3.  ### Result vs exception

    Here it is using a `Result` type, which is a special case of enums. This is
    just like `absl::Status` (https://abseil.io/docs/cpp/guides/status). Several
    advantages vs using exception, IMO.

    -   Exception is part of function signature, but often it's not clear what
        exception would be thrown at where. C++ does have `std::noexcept` but
        that basically it. If it's not `noexcept`, it doesn't mean it would
        raise something either. How do I know what to do with the exception if
        something is really thrown? Not all libraries are like Boost, who
        document all the exceptions that would be thrown (see an example
        [here](https://www.boost.org/doc/libs/1_77_0/doc/html/boost/circular_buffer.html)).
    -   Exception is not allocated on stack (natually, it cannot be on stack).
        It's also not in heap. It's in some magic place determined by the
        compiler. So, tricky memory issues.

4.  Sweet reference. Strong C++ charm. I don't need to explain anything here.

5.  `let` seems to be borrowed from JavaScript / ECMAScript 2015. The type
    annotation style is just like Scala and
    [PEP 484](https://www.python.org/dev/peps/pep-0484/) type hints of Python.
