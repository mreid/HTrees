module Main where

import HTrees

import Data.CSV.Conduit
import Data.List
import Data.Maybe
import qualified  Data.Map.Lazy as Map
import Data.Vector ((!), Vector, fromList)
import System.Environment (getArgs)

-- Main
main = do 
  args <- getArgs 
  let filename = head args

  -- Read and parse the data set
  putStr $ "Reading " ++ filename ++ "... "
  dataset <- readDataset filename
  putStrLn $ (show . length . examples $ dataset) ++ " samples."
  
  -- Construct a tree
  putStr "Builing tree..."
  let config = Config { 
    maxDepth = 30, 
    minNodeSize = 20, 
    leafModel = majModel,  
    stat = Stat { aggregator = toHistogram, summary = entropy } }
  let tree = buildWith config dataset
  putStrLn "Done!"
  putStrLn $ show tree

  -- Evaluation the tree on the training set
  putStrLn "Evaluating ..."
  let evaluation = evaluate zeroOneLoss tree (examples dataset)
  putStrLn $ "MSE = " ++ show evaluation

-- Reads in the "SSV" (semi-colon separated) file and turn it into data
readDataset filename = do 
  csv <- readCSVFile defCSVSettings { csvSep = ';' } filename

  let (names, instances) = parseInstances csv
  let keys = delete "quality" names
  let attrs = [ Attr k (! (fromJust . elemIndex k $ names)) | k <- keys ]
  let target = round . (! (fromJust . elemIndex "quality" $ names)) 
  
  return $ DS attrs (makeExamplesWith target instances)
  
makeExamplesWith :: (Instance -> l) -> [Instance] -> [Example l]
makeExamplesWith f is = [ Ex i (f i) | i <- is ]

--------------------------------------------------------------------------------
-- Data IO and other wrangling

parseInstances :: [MapRow String] -> ([String], [Vector Double])
parseInstances rows = (names, instances) 
  where
    names = Map.keys . head $ rows
    instances = map (makeInstance names) rows

makeInstance :: [String] -> MapRow String -> Instance
makeInstance names row = fromList . map (makeValue . fromJust . flip Map.lookup row) $  names

makeValue :: String -> Double
makeValue s = case reads s :: [(Double, String)] of
  [(v, "")]   -> v
  _           -> error $ "Could not parse '" ++ s ++ "' as Double"

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

