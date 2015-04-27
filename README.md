# HTrees

The code here is an attempt to implement a very basic form of decision tree 
learning in Haskell.

Apart from split ordering, there is no attempt to make things efficient and
there is currently no pruning or other standard tricks to improve the 
learning performance.

## Building and using HTrees

The `src/run.hs` file is what sets up, runs, and evaluates a simple decision 
tree learner on a data set supplied via the command line.

To build this program, use:

	cabal build run

This will create an executable in `dist/build/run/run`. I usually make a 
symbolic link to it from the top level directory:

	ln -s dist/build/run/run run

Then a tree can be trained and evaluated like so:

	./run data/winequality-red.csv

The resulting utput shows the tree that was built and its mean square error
on the traing set.

## Comparison with `rpart`

The R library `rpart` can also be used to build and evaluate decision trees.
For comparison, here is how to build a tree using `rpart`:

	library(rpart)
	wine <- read.csv('data/winequality-red.csv',sep=';')
	fit <- rpart(quality ~ ., data = wine)
	summary(fit)

