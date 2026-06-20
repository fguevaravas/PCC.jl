# procedure to generate: screenshot of font in
# https://www.dafont.com/5x5-pixel.charmap?f=0 and used chaptgpt with
# prompt: "The above image contains 5x5 bitmap representation of the
# numbers 0-9. Output as a 5x5x10 tensor in Julia made of only 0 or 1
# entries"
digits = zeros(5,5,10)
digits[:, :, 1] = [ 
    0 1 1 1 0
    1 0 0 0 1
    1 0 0 0 1
    1 0 0 0 1
    0 1 1 1 0 ]
digits[:, :, 2] = [ 
    0 0 1 0 0
    0 1 1 0 0
    0 0 1 0 0
    0 0 1 0 0
    0 1 1 1 0 ]
digits[:, :, 3] = [ 
    0 1 1 1 0
    1 0 0 0 1
    0 0 0 1 0
    0 0 1 0 0
    1 1 1 1 1 ]
digits[:, :, 4] = [
    1 1 1 1 0
    0 0 0 0 1
    0 0 1 1 0
    0 0 0 0 1
    1 1 1 1 0 ]
digits[:, :, 5] = [ 
    0 0 0 1 0
    0 0 1 1 0
    0 1 0 1 0
    1 1 1 1 1
    0 0 0 1 0 ]

digits[:, :, 6] = [
    1 1 1 1 1
    1 0 0 0 0
    1 1 1 1 0
    0 0 0 0 1
    1 1 1 1 0 ]

digits[:, :, 7] = [ 
    0 1 1 1 0
    1 0 0 0 0
    1 1 1 1 0
    1 0 0 0 1
    0 1 1 1 0 ]

digits[:, :, 8] = [
    1 1 1 1 1
    0 0 0 1 0
    0 0 1 0 0
    0 1 0 0 0
    0 1 0 0 0 ]

digits[:, :, 9] = [
    0 1 1 1 0
    1 0 0 0 1
    0 1 1 1 0
    1 0 0 0 1
    0 1 1 1 0 ]

digits[:, :, 10] = [
    0 1 1 1 0
    1 0 0 0 1
    0 1 1 1 1
    0 0 0 0 1
    0 1 1 1 0 ]
