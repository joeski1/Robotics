Matt:
- maybe don't update every single local map, instead only if the cost is low.
  Maybe generate local maps regularly but update the global map less frequently
- look into parallel line matching
- 230 probabilistic robotics
- more frequent sufficient movement
- different stitching cost function
- face detection
- Erlang graphviz
- auto-expanding global map
- different resolution local and global map
- prevent stitching from guessing that robot hasn't moved?
- Precomputed the indices of the cells that each laser reading intersects
- initialise outside range to -1
- -1 is ignored during stitching right?
- stitching allow side to side search, but penalise forward and backward variation
- image registration problem
- put white pixel in top left to see where stitching
- sum squared difference is sensitive to noise
- https://www.google.co.uk/url?sa=t&source=web&rct=j&url=http://www2.ic.uff.br/iwssip2010/Proceedings/nav/papers/paper_189.pdf&ved=0ahUKEwiI4reyrNTQAhUiIcAKHcNADAsQFggaMAA&usg=AFQjCNGJklhpp8FTxtrUFOyB3h9fvW84dw&sig2=L0PE_9xS0odsTBBgvmx4lQ
- maybe don't update the global map if there are multiple viable stitches which
  are different, or if the match isn't great for any

image cost function
- SIFT
- Zero Mean Normalized Cross-Correlation
- Digital Image Correlation
- http://www.ncorr.com/index.php/dic-algorithms
- have the local map with higher resolution so can move in finer steps and interpolate
- denoise before apply cost function?

Michael:
- stop movment.py from crashing
- down sample global map

Joe, Nishan
- maybe more stitching

charlie:
- check update

test performance of neural network
- expert generated target local map
- compare with NN and AABB
- compare with navstack local maps




Done
- face detection on the map
