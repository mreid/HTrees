module Main where

import HTrees

import Data.CSV.Conduit
import Data.List
import Data.Maybe
import qualified  Data.Map.Lazy as Map
import Data.Vector ((!), Vector, fromList)

-- Main
main = do 
  dataset <- readDataset
  -- Wine data from UCI uses semi-colon separated columns
  putStrLn "Builing..."
  let tree = build 1 dataset
  putStrLn "Done!"
  -- let applied = map stump instances

  putStrLn $ show $ predictWith tree (head . map inst . examples $ dataset)
  -- return $ (dataset, build  1 dataset)

readDataset = do 
  csv <- readCSVFile defCSVSettings { csvSep = ';' } "data/winequality-red.csv" 

  let (names, instances) = parseInstances csv
  let keys = delete "quality" names
  let attrs = [ (! (fromJust . elemIndex k $ names)) | k <- keys ]
  let target = (! (fromJust . elemIndex "quality" $ names))
  
  return $ DS attrs (makeExamplesWith target instances)
  
--------------------------------------------------------------------------------
-- Data IO and other wrangling

parseInstances :: [MapRow String] -> ([String], [Vector Double])
parseInstances rows = (names, instances) 
  where
    names = Map.keys . head $ rows
    instances = map (makeInstance names) rows

makeInstance :: [String] -> MapRow String -> Vector Double
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

