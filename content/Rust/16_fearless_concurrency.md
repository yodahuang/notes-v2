---
share: true
---
# Fearless concurrency

The notes reflect the topics in
[Chapter 16](https://doc.rust-lang.org/book/ch16-00-concurrency.html)

## Thread

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];

    let handle = thread::spawn(move || { // (1)
        println!("Here's a vector: {:?}", v);
    });

    handle.join().unwrap();
}
```

1. That `move` is C++ 14's move capture, but for all the use value used there.
   If `move` is not used here, a compiler error. Sweet.

## Message Passing

So this is from Go (which has super good performance for parallel code). Note
that there's no serialization / de-serialization happening here. The performance
penalty is just some system calls. Let's see some examples.

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    let tx1 = tx.clone();
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals { // (1)
            tx1.send(val).unwrap(); // (2)
            thread::sleep(Duration::from_secs(1));
        }
    });

    thread::spawn(move || {
        let vals = vec![
            String::from("more"),
            String::from("messages"),
            String::from("for"),
            String::from("you"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }
}
```

1. Replacing `vals` with `&vals` would not work. This is because the lifetime of
   the `vals` should be as long as `tx1`, as it will be sent. Recall that
   there's no serialization going on. So the vector should be there for as long
   as someone's receiving it.
2. `val` is being moved here. This is fine with the vector because **all** the
   value in it is moved.

## Shared-State Concurrency

We got our old friend mutex. Conditional variable is not covered, but they are
there in standard library. The mutex here is more like `std::scoped_lock`.
What's more, it's a templated pointer-like thing. That means every mutex covers
exactly one object.

An easy example:

```rust
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);

    {
        let mut num = m.lock().unwrap();
        *num = 6;
    }

    println!("m = {:?}", m);
}
```

But this mutex can be moved only once. And we should not pass raw reference to
value into thread. That's obviously dangerous. So we want, again, a shared
pointer. It can be moved and can have multiple owner.

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter); // (1)
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();

            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

1. The `Arc` here is the thread-safe version of `Rc` covered in Chapter 15.

## Extensible Concurrency with the Sync and Send Traits

`Send` trait means an object can be sent via message passing safely. `Sync`
means it can be accessed (via shared memory) from multiple threads easily.
Implementing it ourselves requires `unsafe`. By default all the primitive value
in Rust already implement these, and the struct containing only these are fine
too.

## Some third party crates

So the standard library is not enough. I found more third party crates:

-   [flume](https://github.com/zesterer/flume): Faster channel.
-   [crossbeam](https://github.com/crossbeam-rs/crossbeam): Data structures and
    more.
-   [rayon](https://github.com/crossbeam-rs/crossbeam): Work stealing
    parallelism. It provides a nice
    [blog post](https://smallcultfollowing.com/babysteps/blog/2015/12/18/rayon-data-parallelism-in-rust/)
    to explain the concept of work stealing. Seems suitable for embarrassingly
    parallel stuff.
