{-# LANGUAGE RankNTypes #-}
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
-- AUTHOR: Mark Reid <mark.reid@gmail.com>
-- CREATED: 2013-02-26
module Main where

import            Control.Arrow ((***))
import            Control.Monad (join)
import            Data.CSV.Conduit
import            Data.Function (on)
import            Data.List (delete, minimumBy, partition)
import qualified  Data.Map.Lazy as Map
import            Data.Maybe (fromJust, mapMaybe)
import            GHC.Exts (groupWith)

--------------------------------------------------------------------------------
-- Main
main = do 
  -- Wine data from UCI uses semi-colon separated columns
  csv <- readCSVFile defCSVSettings { csvSep = ';' } "data/winequality-red.csv" 

  let instances = take 400 $ map parseInstance csv
  let keys = delete "quality" (Map.keys $ head instances) 
  let attrs = [ fromJust . Map.lookup k | k <- keys ]
  let target = fromJust . Map.lookup "quality"
  let dataset = DS attrs (makeExamplesWith target instances)
  putStrLn "Builing..."
  let tree = build 1 dataset
  putStrLn "Done!"
  -- let applied = map stump instances

  putStrLn $ show $ predictWith tree (head instances)
  -- return $ (dataset, build  1 dataset)

--------------------------------------------------------------------------------
-- Data and prediction
-- An example is an instance labelled with a double
type Label = Double
type Value = Double
data Example a = Ex { inst :: a, label :: Label } deriving Show

makeExamplesWith :: (a -> Label) -> [a] -> [Example a]
makeExamplesWith f is = [ Ex i (f i) | i <- is ]

type Attribute a = (a -> Value)

-- A data set is a collection of examples and attributes
data DataSet a = DS {
  features :: [Attribute a],
  examples :: [Example a]
} 

size :: DataSet a -> Int
size = length . examples

instances :: DataSet a -> [a]
instances = map inst . examples

labels :: DataSet a -> [Label]
labels = map label . examples

-- A decision tree consists of internal nodes which split and leaf nodes
-- which make predictions
type Model a = (a -> Label)
data Tree a = Node (a -> Tree a) | Leaf (Model a)

-- Simple prediction with a tree involves running an instance down to a leaf
-- and predicting with the model found there.
predictWith :: Tree a -> a -> Label
predictWith (Leaf f) x      = f x
predictWith (Node branch) x = predictWith (branch x) x

-- Building a tree involves either: 
--  A) Finding a good split for the current examples and refining the tree
--  B) Fitting a model to the examples and returning a leaf
build :: Int -> DataSet a -> Tree a
build depth ds
  | depth > 1 || size ds < 5   = Leaf $ meanModel (examples ds)
  | otherwise                  = Node $ (build (depth + 1) . splitOn best ds)
  where
    best = findBestSplit variance ds

-- Creates a function that maps instances to a 
-- splitOn :: Split a -> DataSet a -> a -> Tree a
splitOn :: Ord k => (a -> k) -> DataSet a -> a -> DataSet a
splitOn split ds i = (DS fs) . Map.findWithDefault [] (split i) $ exSplits
  where
    exSplits = parts split exs
    exs = examples ds 
    fs  = features ds

-- parts :: Ord k => [Example a] -> Split a -> Map.Map k [Example a]
-- parts :: Split a -> [Example a] -> Map.Map k [Example a]
parts :: Ord k => (a -> k) -> [Example a] -> Map.Map k [Example a]
parts split = foldr add Map.empty 
  where
    add ex = Map.insertWith (++) (split $ inst ex) [ex]

--------------------------------------------------------------------------------
-- Splits

-- A Split is a function that returns LT, GT or EQ
type Split a = a -> Bool

-- Build all possible splits for a given data set
allSplits :: DataSet a -> [Split a]
allSplits ds = [makeSplit attr v | attr <- features ds, v <- values attr ds]
  where
    values attr = valuesFor attr 

makeSplit :: Attribute a -> Value -> Split a
makeSplit attr threshold = (< threshold) . attr

valuesFor :: Attribute a -> DataSet a -> [Value]
valuesFor attr = map (attr . inst) . examples

-- Region evaluation
type ImpurityMeasure = [Label] -> Double

-- Computes the variance of the given data set 
variance :: ImpurityMeasure
variance xs = sum . map ((^2) . (subtract (mean xs))) $ xs

-- Given several label lists returns a size weighted average impurity
quality :: ImpurityMeasure -> [[Label]] -> Double
quality measure [] = read "Infinity" 
quality measure ls = sum (zipWith (*) qualities sizes) / size ls
  where
    size      = fromIntegral . length 
    qualities = map measure ls
    sizes     = map size ls

-- Computes the quality of a split applied to a particular data set
splitQuality :: ImpurityMeasure -> DataSet a -> Split a -> Double
splitQuality measure ds split = quality measure labelSets
  where
    exampleLists = groupWith (split . inst) (examples ds)
    labelSets = map (map label) exampleLists

-- Get best split for the given data set as assessed by the impurity measure
findBestSplit :: ImpurityMeasure -> DataSet a -> Split a
findBestSplit im ds = minimumBy (compare `on` splitQuality im ds) (allSplits ds)

--------------------------------------------------------------------------------
-- Models

-- Constructs a constant model which returns the mean of the examples
meanModel :: [Example a] -> Model a
meanModel xs = const (mean . map label $ xs)

-- Compute the mean of a list of numbers
mean :: [Double] -> Double
mean xs = (sum xs) / (fromIntegral . length $ xs)

--------------------------------------------------------------------------------
-- Data IO and other wrangling

parseInstance :: MapRow String -> Map.Map String Double
parseInstance = Map.map makeValue 

makeValue :: String -> Double
makeValue s = case reads s :: [(Double, String)] of
  [(v, "")]   -> v
  _               -> error $ "Could not parse '" ++ s ++ "' as Double"

-- Wine data set columns:
-- "fixed acidity";
-- "volatile acidity";
-- "citric acid";
-- "residual sugar";
-- "chlorides";
-- "free sulfur dioxide";
-- "total sulfur dioxide";
-- "density";
-- "pH";
-- "sulphates";
-- "alcohol";
-- "quality"

