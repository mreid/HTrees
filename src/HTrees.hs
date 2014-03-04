-- {-# LANGUAGE TypeSynonymInstances, FlexibleInstances #-}
-- An attempt at writing a simple decision tree learner, with an emphasis
-- on expression and generality rather than efficiency.
--
-- As much as possible, I am using the same conceptual framework as PSI:
--   * Instances are abstract
--   * Attributes map instances to immutable values
--   * Supervised examples can be expressed via:
--      • A collection of instances
--      • A source attribute mapping instances to some feature representation
--      • A target attribute mapping instances to some target value
-- 
-- In order to create decision trees from examples, we require that a 
-- collection of possible projections be provided along with the instances.
-- These are attributes that map instances to orderable (e.g., continuous) or 
-- equality-testable values (e.g., categorical).
--
-- In the case of the wine dataset from UCI, the source attribute can return
-- a Wine value which has projections 
--    fixedAcidity :: Wine -> Double
--    volatileAcidity :: Wine -> Double
--
-- References:
--  * An alternative implementation: https://github.com/ajtulloch/haskell-ml
--
-- AUTHOR: Mark Reid <mark.reid@gmail.com>
-- CREATED: 2013-02-26
module HTrees where

import            Control.Arrow ((***))
import            Control.Monad (join)
import            Data.Monoid   (Monoid, mempty, mappend)
import            Data.Foldable (foldMap)
import            Data.Function (on)
import            Data.List (minimumBy, partition, sortBy)

import qualified  Data.Map.Lazy as Map
--------------------------------------------------------------------------------
-- Data and prediction
-- An example is an instance labelled with a double
type Label = Double
type Value = Double
type Attribute a = (a -> Value)

data Example a = Ex { inst :: a, label :: Label } deriving Show

-- A data set is a collection of examples and attributes
data DataSet a = DS { features :: [Attribute a], examples :: [Example a] } 

size :: DataSet a -> Int
size = length . examples

-- A decision tree consists of internal nodes which split and leaf nodes
-- which make predictions
type Model a = (a -> Label)
data Tree a = Node (Split a) (Tree a) (Tree a) | Leaf (Model a)

-- Simple prediction with a tree involves running an instance down to a leaf
-- and predicting with the model found there.
predictWith :: Tree a -> a -> Label
predictWith (Leaf f) x = f x
predictWith (Node (Split attr val) left right) x 
  | (attr x < val)  = predictWith left x
  | otherwise       = predictWith right x

-- Building a tree involves either: 
--  A) Finding a good split for the current examples and refining the tree
--  B) Fitting a model to the examples and returning a leaf
build :: Int -> DataSet a -> Tree a
build depth ds@(DS fs exs)
  | depth > 1 || size ds < 5   = Leaf $ meanModel exs
  | otherwise                  = Node split left right
  where
    split = findBestSplit variance ds
    test = (< threshold split) . (attribute split). inst
    (left, right) = join (***) (build (depth+1) . DS fs) $ partition test exs

--------------------------------------------------------------------------------
-- Impurity Measures

-- A statistic is a Monoid with an associated ``weight'' (e.g., # of samples)
-- and functions to put samples in and get summaries out. 
class Monoid a => Statistic a where
  fromSample :: Double -> a
  toDouble   :: a -> Double
  weight     :: a -> Double

-- Variance works by keep a running some of values and a sum of squared values
-- (Note: This is *not* a numerically stable implementation)
newtype Variance = Var (Double, Double, Double)
variance = fromSample :: Label -> Variance

instance Monoid Variance where
  mempty = Var (0,0,0)
  mappend (Var (nx,sx,sx2)) (Var (ny,sy,sy2)) = Var (nx+ny,sx+sy,sx2+sy2)

instance Statistic Variance where
  fromSample v            = Var (1, v, v**2)
  weight (Var (n,_,_))    = n
  toDouble (Var (n,s,s2)) = s2/n - (s/n)**2

--------------------------------------------------------------------------------
-- Splits

-- A Split is puts each instance into one of two classes by testing 
-- an attribute against a threshold.
data Split a = Split { attribute :: Attribute a, threshold :: Value } 

-- Build all possible splits for a given data set
-- allSplits :: DataSet a -> [Split a]
allSplits ds = [Split attr v | attr <- features ds, v <- values attr]
  where
    values attr = map (attr . inst) . examples $ ds

quality :: Statistic s => (Value, s) -> (Value, s) -> (Value,Double)
quality (v,left) (v',right) = (v,(weight left) * (toDouble left) + (weight right) * (toDouble right))

-- Get best split for the given data set as assessed by the impurity measure
findBestSplit :: Statistic s => (Label -> s) -> DataSet a -> Split a
findBestSplit stat ds = fst . minimumBy (compare `on` snd) $ perAttr
  where
    perAttr = map (bestSplitFor (examples ds) stat) (features ds)

bestSplitFor :: Statistic s => [Example a] -> (Label -> s) -> Attribute a -> (Split a, Double)
bestSplitFor exs stat attr = (Split attr val, score)
  where
    sorted = sortBy (compare `on` (attr . inst)) exs
    accum  = foldMap (f stat attr [])
    scores = zipWith quality (accum sorted) (accum . reverse $ sorted)
    (val, score) = minimumBy (compare `on` snd) scores

f :: Statistic s => (Label -> s) -> Attribute a -> [(Value, s)] -> Example a ->  [(Value, s)]
f stat attr [] ex = [(attr . inst $ ex, stat . label $ ex)]
f stat attr pairs ex = (attr . inst $ ex, (stat . label $ ex) `mappend` (snd . head $ pairs)) : pairs

--------------------------------------------------------------------------------
-- Models

-- Constructs a constant model which returns the mean of the examples
meanModel :: [Example a] -> Model a
meanModel xs = const (mean . map label $ xs)

-- Compute the mean of a list of numbers
mean :: [Double] -> Double
mean xs = (sum xs) / (fromIntegral . length $ xs)

