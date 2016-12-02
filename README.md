Atlas Viewer for the Allen Institute Common Coordinate Framework
================================================================

Features:
---------

* Display CCF atlas data sliced at arbitrary angles
* Color atlas regions based on their anatomical labels or by cortical layer
* Displays anatomical label of the atlas region under the mouse pointer


Requirements:
-------------

* Python 2.7
* PyQt4
* PyQtGraph
* numpy
* pynrrd
* h5py 


Setup:
------

The viewer requires three pieces of data that can be downloaded from the Allen Institute website:

* Atlas data: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/
* Label data: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2015/
* Ontology: http://api.brain-map.org/api/v2/structure_graph_download/1.json

More information about the atlas data and ontology files is available at:
* http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas
* http://help.brain-map.org/display/api/Downloading+an+Ontology%27s+Structure+Graph
* http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-StructuresAndOntologies

Start the viewer from the command prompt:

```
$ python viewer.py
```

The first time the viewer runs, it will ask for the location of the three files listed above.
Data is then converted into a format that is more memory- and processor-efficient; this process can take
several minutes depending on the resolution of the atlas/label files you select (the file names indicate the
voxel size for each data set; for example, `annotation_10.nrrd` contains the CCF label data with 10 um voxels).



