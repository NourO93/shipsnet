// dependencies
const https = require('https');
const querystring = require('querystring');
const fs = require('fs');


/**
  This is our AND filter specifying:
    1) a date range, requesting imagery from the beginning of 2017
    2) a black_fill parameter specifying that we want imagery that is at least 85 percent covered
    3) images that can be downloaded
**/
const postData = JSON.stringify(
{
  "item_types": [ 'Sentinel2L1C' /**,"Landsat8L1G"**/ ],
      "filter": {
          "type": "AndFilter",
          "config": [
              {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": {
                   "type": "Polygon",
                    "coordinates": [
                      [
                        [
                          -119.08905029296874,
                          33.98664113654014
                        ],
                        [
                          -118.50677490234375,
                          33.98664113654014
                        ],
                        [
                          -118.50677490234375,
                          34.168635904722734
                        ],
                        [
                          -119.08905029296874,
                          34.168635904722734
                        ],
                        [
                          -119.08905029296874,
                          33.98664113654014
                        ]
                      ]
                    ]              
                }
              },

              {
                  "type": "DateRangeFilter",
                  "field_name": "acquired",
                  "config": {
                      "gte": "2017-01-01T00:00:00Z"
                  }
              },
              {
                "type": "RangeFilter",
                "field_name": "black_fill",
                "config": {
                    "lte": 0.15
                }
              },
              {
                "type": "PermissionFilter",
                "config": ["assets:download"]
              }
          ]
      }
});

// API key and construction of the header
const username = '<api_key>'

var head = {
  'Authorization':  'Basic '  + new Buffer(username + ':').toString('base64'),
  'Content-Type':   'application/json' 
}

// Specify the main request structure
const options = {
  hostname: "api.planet.com",
  port:443,
  path:"/data/v1/quick-search",
  method: 'POST',
  headers: head  
}

// setup the request handlers
const req = https.request(options, (res) => {
  
  console.log(`STATUS: ${res.statusCode}`);
  console.log(`HEADERS: ${JSON.stringify(res.headers)}`);
  
  res.setEncoding('utf8');
  var bufferResponse = '';

  // buffer the response
  res.on('data', (chunk) => {
    
    var dataResponse = chunk; 
    bufferResponse += chunk;
    
  });
  
  // now that we're done let's write the data to a file and process the data a little
  res.on('end', () => {
    
    console.log(bufferResponse)
    writeToFile(bufferResponse)
    processResults( bufferResponse);

  });
});

req.on('error', (e) => {
  console.error(`problem with request: ${e.message}`);
});

// write data to request body
req.write(postData);
req.end();

// wrie the data to a file
function writeToFile(dataResponse){
  fs.appendFile("outdata.json", dataResponse, 'utf8', (err) => {
      if (err) throw err;
      console.log('The file has been saved!');
    });

}

// just count the results, obviously we could do a lot more here!
function processResults(response){
  var obj = JSON.parse(response)
  
  if(obj.features.length !== undefined){
    console.log(obj.features.length)
  }
  
}