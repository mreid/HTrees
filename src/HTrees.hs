{-# LANGUAGE FlexibleInstances #-}
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

import            Control.Applicative ((<$>), (<*>), liftA2)
import            Control.Arrow       ((***))
import            Control.Monad       (join)
import            Data.Foldable       (foldMap)
import            Data.Function       (on)
import            Data.List           (minimumBy, partition, sortBy)
import            Data.Monoid         ((<>), Monoid, mempty, mappend)
import            Data.Vector         (Vector)

import qualified  Data.Map.Lazy as Map

--------------------------------------------------------------------------------
-- Data and prediction
-- An example is an instance labelled with a double
type Value = Double
type Instance = Vector Double
data Example l = Ex { inst :: Instance, label :: l } deriving Show

-- A data set is a collection of examples and attributes
data Attribute = Attr { name :: String, project :: Instance -> Value }
instance Show Attribute where show (Attr name' _) = name'

projectEx :: Attribute -> Example l -> Value
projectEx attr' = project attr' . inst

data DataSet l = DS { attributes :: [Attribute], examples :: [Example l] }
  deriving Show

-- A decision tree consists of internal nodes which split and leaf nodes
-- which make predictions
data Tree l = Node Split (Tree l) (Tree l) | Leaf (Model l)
instance Show (Tree l) where show t = showTree 0 t

showTree :: Int -> Tree l -> String
showTree d (Node s l r) = indent d (show s) ++ ":\n" ++
   indent d (showTree (d+1) l) ++ "\n" ++
   indent d (showTree (d+1) r)
showTree d (Leaf m)     = indent d (show m)

indent :: Int -> String -> String
indent d = (replicate d ' ' ++)

-- Simple prediction with a tree involves running an instance down to a leaf
-- and predicting with the model found there.
predictWith :: Tree l -> Instance -> l
predictWith (Leaf m) x = (fn m) x
predictWith (Node (Split attr' val _) left right) x
  | ((project attr') x < val) = predictWith left x
  | otherwise                 = predictWith right x

-- Configuration variables for tree building kept here.
data Config l p s = Config {
  maxDepth    :: Int,
  minNodeSize :: Int,
  leafModel   :: [Example l] -> Model p,
  stat        :: Stat l s
}

defRegConfig :: Config Double Double Moments
defRegConfig    = Config 32 10 meanModel (Stat toMoment variance)

defClassConfig :: Config Int Int Histogram
defClassConfig  = Config 32 10 majModel (Stat toHistogram entropy)

-- Check whether tree building should stop by seeing whether maximum depth
-- has been reached or the current dataset is too small.
isFinished :: (Foldable f, Foldable g) => Config a b c -> (f x, g y) -> Bool
isFinished (Config maxDepth' minNodeSize' _ _) (leftExs, rightExs)
  = (maxDepth' == 0)
    || (length leftExs  <= minNodeSize')
    || (length rightExs <= minNodeSize')

-- Building a tree involves either:
--  A) Finding a good split for the current examples and refining the tree
--  B) Fitting a model to the examples and returning a leaf
buildWith :: (Aggregate a, Num l) => Config l p a -> DataSet l -> Tree p
buildWith config ds
  | isFinished config splitExs  = Leaf $ (leafModel config) (examples ds)
  | otherwise                   = Node split left right
  where
    split     = findBestSplit (stat config) ds
    test      = (<= value split) . projectEx (attr split)
    splitExs  = partition test (examples ds)
    newConfig = config { maxDepth = maxDepth config - 1 }
    attrs     = attributes ds
    (left, right) = join (***) (buildWith newConfig . DS attrs) splitExs

-- Evaluate the predictions made by a tree
evaluate :: Loss l p -> Tree p -> [Example l] -> Double
evaluate loss tree = mean . map (loss <$> label <*> (predictWith tree . inst))

-- The value of loss p a is the penalty for predicting p when true label is l.
type Loss l p = l -> p -> Double
squareLoss :: Loss Double Double
squareLoss v v' = (v - v')**2

zeroOneLoss :: Loss Int Int
zeroOneLoss v v' = if v == v' then 0 else 1
--------------------------------------------------------------------------------
-- Impurity Measures

-- A Stat has a function for taking a value to an Aggregate and summarising
-- an Aggregate of the same type
data Stat l s = Stat { aggregator :: l -> s, summary :: s -> Double }

-- Combine two statistics by taking their weighted average
mergeWith :: Aggregate a => Stat l a -> a -> a -> Double
mergeWith stat' a a'
  = ( (size a) * (summariser a) + (size a') * (summariser a') ) / both
  where
    summariser = summary stat'
    both = (size a) + (size a')

-- Returns the variance from a moment aggregation
-- (Note: This is *not* a numerically stable way to do things)
variance :: Moments -> Double
variance (Mom (n, s, s2)) =  s2/n - (s/n)**2

-- Returns the entropy of a histogram
entropy :: Histogram -> Double
entropy (Hist (n, fs)) = - sum (zipWith (*) ps (map log2 ps))
  where
    ps   = map (/ n) . Map.elems $ fs
    log2 = logBase 2
--------------------------------------------------------------------------------
-- Aggregators are Monoids that collect values into summaries with sizes
class Monoid s => Aggregate s where
  size      :: s -> Double

-- Collects the fist and second moments of the continuous values it sees
newtype Moments = Mom (Double, Double, Double)
toMoment :: Double -> Moments
toMoment d = Mom (1,d,d*d)

instance Aggregate Moments where size (Mom (n, _, _)) = n
instance Monoid Moments where
  mempty = Mom (0,0,0)
  mappend (Mom (n,s,s2)) (Mom (n',s',s2')) = (Mom (n+n',s+s',s2+s2'))

-- Creates a histogram of the discrete values it sees
newtype Histogram = Hist (Double,Map.Map Int Double)
toHistogram :: Int -> Histogram
toHistogram i = Hist (1, Map.singleton i 1)

instance Aggregate Histogram where size (Hist (_,fs)) = Map.foldr (+) 0 fs
instance Monoid Histogram where
  mempty = Hist (0,Map.empty)
  mappend (Hist (n1, fs1)) (Hist (n2, fs2))
    = Hist (n1+n2, Map.unionWith (+) fs1 fs2)

--------------------------------------------------------------------------------
-- Splits

-- A Split is puts each instance into one of two classes by testing
-- an attribute against a threshold.
data Split = Split { attr :: Attribute,  value :: Value, score :: Double}
instance Show Split where
  show (Split (Attr name' _) v q)
    = name' ++ " <= " ++ (show v) ++ " (Impurity: " ++ show q ++ ")"

-- Build all possible splits for a given data set
allSplits :: DataSet l -> [Split]
allSplits ds = [Split attr' v infty | attr' <- attributes ds, v <- values attr']
  where
    infty = read "Infinity" :: Double
q    values attr' = map (projectEx attr') . examples $ ds

-- Get best split for the given data set as assessed by the impurity measure
findBestSplit :: (Aggregate a, Num l) => Stat l a -> DataSet l -> Split
findBestSplit stat' ds = minimumBy (compare `on` score) $ bestPerAttr
  where
    bestPerAttr = map (bestSplit (examples ds) stat') $ attributes ds

-- Get the best split and its score for the given statistic and attribute
bestSplit :: Aggregate a => [Example l] -> Stat l a -> Attribute -> Split
bestSplit exs stat' attr' = minimumBy (compare `on` score) pairs
  where
    -- Sort examples by values for attribute
    sorted    = sortBy (compare `on` projectEx attr') exs
    labels    = map label sorted
    -- Roll together stats from front and back of sorted list of examples
    -- Note: backwards list is one shorter since forward splits test "<= v"
    forwards  = foldr accum [] labels
    backwards = reverse . foldr accum [] . reverse . tail $ labels
    scores    = zipWith (mergeWith stat') forwards backwards
    pairs     = zipWith (Split attr') (map (projectEx attr') sorted) scores
    -- Accumulator
    accum l [] = [(aggregator stat') l]
    accum l vs = ((aggregator stat') l `mappend` head vs) : vs

--------------------------------------------------------------------------------
-- Models

data Model l = Model { desc :: String
                     , fn :: (Instance -> l)
                     , input :: [Example l] }
instance Show (Model l) where
  show (Model d _ exs) = d ++ " (Samples: " ++ show (length exs) ++ ")"

-- Constructs a constant model which returns the mean of the examples
meanModel :: [Example Double] -> Model Double
meanModel xs = Model ("Predict " ++ show v) (const v) xs
  where
    v = (mean . map label $ xs)

-- Compute the mean of a list of numbers
mean :: [Double] -> Double
mean xs = (sum xs) / (fromIntegral . length $ xs)

-- Compute the majority class model from a collection of examples
majModel :: [Example Int] -> Model Int
majModel xs = Model ("Predict " ++ show v) (const v) xs
  where
    Hist (_, fs) = foldMap (toHistogram . label) xs
    v = fst . Map.findMax $ fs
