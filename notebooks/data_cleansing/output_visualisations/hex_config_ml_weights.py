config = {
    "version": "v1",
    "config": {
        "visState": {
            "layers": [
                {
                    "id": "5wto52g",
                    "type": "geojson",
                    "config": {
                        "dataId": "density",
                        "label": "density",
                        "color": [
                            18,
                            147,
                            154
                        ],
                        "highlightColor": [
                            252,
                            242,
                            26,
                            255
                        ],
                        "columns": {
                            "geojson": "geometry"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "opacity": 0.8,
                            "strokeOpacity": 0.8,
                            "thickness": 0.5,
                            "strokeColor": None,
                            "colorRange": {
                                "name": "Global Warming",
                                "type": "sequential",
                                "category": "Uber",
                                "colors": [
                                    "#5A1846",
                                    "#900C3F",
                                    "#C70039",
                                    "#E3611C",
                                    "#F1920E",
                                    "#FFC300"
                                ]
                            },
                            "strokeColorRange": {
                                "name": "ColorBrewer RdYlGn-6",
                                "type": "diverging",
                                "category": "ColorBrewer",
                                "colors": [
                                    "#1a9850",
                                    "#91cf60",
                                    "#d9ef8b",
                                    "#fee08b",
                                    "#fc8d59",
                                    "#d73027"
                                ],
                                "reversed": True
                            },
                            "radius": 10,
                            "sizeRange": [
                                0,
                                10
                            ],
                            "radiusRange": [
                                0,
                                50
                            ],
                            "heightRange": [
                                0,
                                500
                            ],
                            "elevationScale": 5,
                            "enableElevationZoomFactor": True,
                            "stroked": True,
                            "filled": False,
                            "enable3d": False,
                            "wireframe": False
                        },
                        "hidden": False,
                        "textLabel": [
                            {
                                "field": None,
                                "color": [
                                    255,
                                    255,
                                    255
                                ],
                                "size": 18,
                                "offset": [
                                    0,
                                    0
                                ],
                                "anchor": "start",
                                "alignment": "center"
                            }
                        ]
                    },
                    "visualChannels": {
                        "colorField": None,
                        "colorScale": "quantile",
                        "strokeColorField": {
                            "name": "van",
                            "type": "real"
                        },
                        "strokeColorScale": "quantile",
                        "sizeField": None,
                        "sizeScale": "linear",
                        "heightField": None,
                        "heightScale": "linear",
                        "radiusField": None,
                        "radiusScale": "linear"
                    }
                }
            ],
            "interactionConfig": {
                "tooltip": {
                    "fieldsToShow": {
                        "density": [
                            {
                                "name": "street_object_id",
                                "format": None
                            },
                            {
                                "name": "bike",
                                "format": None
                            },
                            {
                                "name": "bus",
                                "format": None
                            },
                            {
                                "name": "car",
                                "format": None
                            },
                            {
                                "name": "truck",
                                "format": None
                            },
                            {
                                "name": "van",
                                "format": None
                            },
                            {
                                "name": "walk",
                                "format": None
                            }
                        ]
                    },
                    "compareMode": False,
                    "compareType": "absolute",
                    "enabled": True
                },
                "brush": {
                    "size": 0.5,
                    "enabled": False
                },
                "geocoder": {
                    "enabled": False
                },
                "coordinate": {
                    "enabled": False
                }
            },
            "layerBlending": "normal",
            "splitMaps": [],
            "animationConfig": {
                "currentTime": None,
                "speed": 1
            }
        },
        "mapState": {
            "bearing": 0,
            "dragRotate": False,
            "latitude": 51.20280957866893,
            "longitude": 4.407925581174888,
            "pitch": 0,
            "zoom": 12.401918059850553,
            "isSplit": False
        },
        "mapStyle": {
            "styleType": "dark",
            "topLayerGroups": {},
            "visibleLayerGroups": {
                "label": True,
                "road": True,
                "border": False,
                "building": True,
                "water": True,
                "land": True,
                "3d building": False
            },
            "threeDBuildingColor": [
                9.665468314072013,
                17.18305478057247,
                31.1442867897876
            ],
            "mapStyles": {}
        }
    }
}
