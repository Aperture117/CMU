SELECT
    -- Select exact headquater lng,lat and make distance as KM for convinience
    -- Make new cols which is launch site zipcodes
    -- Count all orders as Order_count cols
    LaunchZips.Zip AS launch_zip_code,
    COUNT(Orders.OrderId) AS order_count,
    ROUND(ST_DISTANCE_SPHERE(POINT(LaunchZips.Lng, LaunchZips.Lat), POINT(-71.10253, 42.36224)) / 1000, 2) AS distance_km
FROM
    Zips AS LaunchZips
JOIN
    -- Select Maximum distance 100KM
    Customer ON ST_DISTANCE_SPHERE(POINT(LaunchZips.Lng, LaunchZips.Lat), POINT(-71.10253, 42.36224)) <= 100000 
JOIN
    Orders ON Orders.CustomerId = Customer.CustomerId
JOIN
    Zips AS OrderZips ON OrderZips.Zip = Customer.PostalCode
WHERE
    -- Select Maximum Drone distance as 3KM 
    ST_DISTANCE_SPHERE(POINT(LaunchZips.Lng, LaunchZips.Lat), POINT(OrderZips.Lng, OrderZips.Lat)) <= 3000 
GROUP BY
    LaunchZips.Zip, LaunchZips.Lng, LaunchZips.Lat
HAVING
    -- Select Maximum distance of headquater drone distance(3+3=6 KM) and within 100KM except Headquater
    distance_km BETWEEN 6 AND 100 AND LaunchZips.Zip <> 02139
ORDER BY
    order_count DESC
LIMIT 3;
