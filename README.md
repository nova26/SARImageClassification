# SARImageClassification

Imagery data from satellite can have the same general characteristics as visual imagery but still have specific attributes. 

A satellite radar is working quite the same way as blips on a ship or a radar on an aircraft.
The radar bounces a signal off an object and records the echo, then the data is being translated into an image. 
An object will appear as a bright spot.
This is because the object reflects more radar energy than its surroundings, but strong echoes can come from anything that is solid - land, islands, sea ice, as well as ships and icebergs. The energy reflected to the radar is referred to as backscatter. 
 
 
 
When the radar detects an object, it can't tell an iceberg from a ship or any other solid object.
To find out the answer, the object needs to be analyzed for characteristics like shape, size and brightness.
The area surrounding the object, in our case: the ocean, can also be analyzed or modeled.  
Many things affect the backscatter of the ocean or background area.
High winds will generate a brighter background. Conversely, low winds will generate a darker background. 
The Sentinel-1 satellite is a side looking radar, which means it sees the image area at an angle (incidence angle). 
Generally, the ocean background will be darker at a higher incidence angle.
You also need to consider the radar polarization, which is way the radar transmits and receives energy.
More advanced radars like Sentinel-1, can transmit and receive in the horizontal and vertical planes.
Using this, you can get what is called a dualpolarization image. 
