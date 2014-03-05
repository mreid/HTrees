-- HTrees - Simple tree learning.
--
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
-- References:
--  * An alternative implementation: https://github.com/ajtulloch/haskell-ml
--
-- AUTHOR: Mark Reid <mark.reid@gmail.com>
-- CREATED: 2013-02-26
module HTrees where

import            Control.Applicative ((<$>), (<*>))
import            Control.Arrow       ((***))
import            Control.Monad       (join)
import            Data.Foldable       (foldMap)
import            Data.Function       (on)
import            Data.List           (minimumBy, partition, sortBy)
import            Data.Monoid         (Monoid, mempty, mappend)

import qualified  Data.Map.Lazy as Map

--------------------------------------------------------------------------------
-- Data and prediction
-- An example is an instance labelled with a double
type Value = Double
data Example a l = Ex { inst :: a, label :: l } deriving Show

-- A data set is a collection of examples and attributes
data Attribute a = Attr { name :: String, project :: a -> Value }
instance Show (Attribute a) where show (Attr name _) = name

projectEx :: Attribute a -> Example a l -> Value
projectEx attr = project attr . inst

data DataSet a l = DS { attributes :: [Attribute a], examples :: [Example a l] } 
  deriving Show

-- A decision tree consists of internal nodes which split and leaf nodes
-- which make predictions
data Tree a l = Node (Split a) (Tree a l) (Tree a l) | Leaf (Model a l)
instance Show (Tree a l) where show t = showTree 0 t

showTree :: Int -> Tree a l -> String
showTree d (Node s l r) = indent d (show s) ++ ":\n" ++ 
   indent d (showTree (d+1) l) ++ "\n" ++ 
   indent d (showTree (d+1) r)
showTree d (Leaf m)     = indent d (show m)
indent d = (replicate d ' ' ++) 

-- Simple prediction with a tree involves running an instance down to a leaf
-- and predicting with the model found there.
predictWith :: Tree a l -> a -> l
predictWith (Leaf m) x = (fn m) x
predictWith (Node (Split attr val _) left right) x 
  | ((project attr) x < val)  = predictWith left x
  | otherwise                 = predictWith right x

-- Configuration variables for tree building kept here.
data Config = Config { maxDepth :: Int,  minNodeSize :: Int }
defConfig = Config { maxDepth = 32, minNodeSize = 10 }

-- Check whether tree building should stop by seeing whether maximum depth
-- has been reached or the current dataset is too small.
isFinished (Config maxDepth minNodeSize) (leftExs, rightExs)
  = (maxDepth == 0) 
    || (length leftExs <= minNodeSize) 
    || (length rightExs <= minNodeSize)

-- Building a tree involves either: 
--  A) Finding a good split for the current examples and refining the tree
--  B) Fitting a model to the examples and returning a leaf
buildWith :: Stat s => Config -> (l -> s) -> ([Example a l] -> Model a l) -> DataSet a l -> Tree a l
buildWith config stat leafModel ds
  | isFinished config splitExs  = Leaf $ leafModel (examples ds)
  | otherwise                   = Node split left right
  where
    split     = findBestSplit stat ds
    test      = (<= value split) . projectEx (attr split)
    splitExs  = partition test (examples ds)
    newConfig = config { maxDepth = maxDepth config - 1 }
    attrs     = attributes ds
    (left, right) = join (***) (buildWith newConfig stat leafModel. DS attrs) splitExs

-- Evaluate the predictions made by a tree
evaluate :: Loss l -> Tree a l -> [Example a l] -> Double
evaluate loss tree = mean . map (loss <$> label <*> (predictWith tree . inst))

-- The value of loss p a is the penalty for predicting p when true label is a.
type Loss l = l -> l -> Double
squareLoss :: Loss Double
squareLoss v v' = (v - v')**2

--------------------------------------------------------------------------------
-- Impurity Measures
type Histogram = [(Int,Double)]
newtype Entropy = Ent Histogram
instance Monoid Entropy where
  mempty = Ent []
  mappend (Ent v1) (Ent v2) = Ent (v1++v2)
instance Stat Entropy where
  toDouble (Ent vs) = undefined
  weight (Ent vs) = undefined
entropy :: Int -> Entropy
entropy l = Ent [(l,1)]

-- A statistic is a Monoid with an associated ``weight'' (e.g., # of samples)
-- and functions to put samples in and get summaries out. 
class Monoid s => Stat s where
  toDouble   :: s -> Double
  weight     :: s -> Double

-- Combine two statistics by taking their weighted average
merge :: Stat s => s -> s -> Double
merge s s' = ( (weight s) * (toDouble s) + (weight s') * (toDouble s') ) / both
  where
    both = (weight s) + (weight s')

-- Variance works by keep a running some of values and a sum of squared values
-- (Note: This is *not* a numerically stable implementation)
newtype Variance = Var (Double, Double, Double)
variance v = Var (1, v, v**2)

instance Monoid Variance where
  mempty = Var (0,0,0)
  mappend (Var (nx,sx,sx2)) (Var (ny,sy,sy2)) = Var (nx+ny,sx+sy,sx2+sy2)

instance Stat Variance where
  weight (Var (n,_,_))    = n
  toDouble (Var (n,s,s2)) = s2/n - (s/n)**2

--------------------------------------------------------------------------------
-- Splits

-- A Split is puts each instance into one of two classes by testing 
-- an attribute against a threshold.
data Split a = Split { attr :: Attribute a, value :: Value, score :: Double} 
instance Show (Split a) where 
  show (Split (Attr name _) v q) 
    = name ++ " <= " ++ (show v) ++ " (Impurity: " ++ show q ++ ")"

-- Build all possible splits for a given data set
allSplits :: DataSet a l -> [Split a]
allSplits ds = [Split attr v infty | attr <- attributes ds, v <- values attr]
  where
    infty = read "Infinity" :: Double
    values attr = map (projectEx attr) . examples $ ds

-- Get best split for the given data set as assessed by the impurity measure
findBestSplit :: Stat s => (l -> s) -> DataSet a l -> Split a
findBestSplit stat ds = minimumBy (compare `on` score) $ bestPerAttr
  where
    bestPerAttr = map (bestSplit (examples ds) stat) $ attributes ds

-- Get the best split and its score for the given statistic and attribute
bestSplit :: Stat s => [Example a l] -> (l -> s) -> Attribute a -> Split a
bestSplit exs stat attr = minimumBy (compare `on` score) pairs
  where
    -- Sort examples by values for attribute
    sorted    = sortBy (compare `on` projectEx attr) exs
    labels    = map label sorted
    -- Roll together stats from front and back of sorted list of examples
    -- Note: backwards list is one shorter since forward splits test "<= v"
    forwards  = foldr accum [] labels
    backwards = reverse . foldr accum [] . reverse . tail $ labels
    scores    = zipWith merge forwards backwards
    pairs     = zipWith (Split attr) (map (projectEx attr) sorted) scores
    -- Accumulator 
    accum l [] = [stat l]
    accum l vs = (stat l `mappend` head vs) : vs

--------------------------------------------------------------------------------
-- Models
data Model a l = Model { desc :: String, fn :: (a -> l), input :: [Example a l] }
instance Show (Model a l) where 
  show (Model d m exs) = d ++ " (Samples: " ++ show (length exs) ++ ")"

-- Constructs a constant model which returns the mean of the examples
meanModel :: [Example a Double] -> Model a Double
meanModel xs = Model ("Predict " ++ show v) (const v) xs
  where
    v = (mean . map label $ xs)

-- Compute the mean of a list of numbers
mean :: [Double] -> Double
mean xs = (sum xs) / (fromIntegral . length $ xs)

