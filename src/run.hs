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
  putStrLn "Builing..."
  let config = Config { maxDepth = 30, minNodeSize = 20  }
  let tree = buildWith config variance dataset
  putStrLn $ show tree
  putStrLn "Done!"
  
  putStr "Mean square error: "
  putStrLn $ show $ evaluate squareLoss tree (examples dataset)

readDataset = do 
  csv <- readCSVFile defCSVSettings { csvSep = ';' } "data/winequality-red.csv" 

  let (names, instances) = parseInstances csv
  let keys = delete "quality" names
  let attrs = [ Attr k (! (fromJust . elemIndex k $ names)) | k <- keys ]
  let target = (! (fromJust . elemIndex "quality" $ names))
  
  return $ DS attrs (makeExamplesWith target instances)
  
makeExamplesWith :: (a -> Label) -> [a] -> [Example a]
makeExamplesWith f is = [ Ex i (f i) | i <- is ]

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

