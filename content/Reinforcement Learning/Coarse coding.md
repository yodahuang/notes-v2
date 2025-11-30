Somehow I only see this name in the context of RL. I suspect it just so happens that it's in the famous Richard S.Sutton and Andrew G. Barto book.

This is a way to "expand" the state embedding, or state encoder.

For example, we can convert the 2D coordinate to this representation:
![[coarse_coding.png]]
And we assign each circle a binary value, and we can represent the x location by 3 `1`s for the three circle and `0` for other circles. And it preserves locality. We can also do it by gaussian.