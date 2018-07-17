import numpy as np
from .vrep_files import vrep

from collections import namedtuple


ColorParam = namedtuple('ColorParam', ['rgb', 'mass'])


class ObjectConfig(object):
    """Container for object configuration info to pass through VREP API in the
    expected format. Note that this info is used by a special API call that
    just passes arguments to a Lua function defined in VREP. These arguments are
    required to be lists of items of different datatypes, hence the property
    definitions in this class.

    Parameters:
    -----------
    name : str
        The name of the object
    kind : str
        The kind or type of object (cube, sphere, cylinder, or cone)
    color : str
        The color of the object, used to set RGB and mass
    xyz : list of floats
        The coordinates at which to center the object
    size : list of floats
        The extent of the object along each axis (x,y,z)
    vrep_dummy : str (default 'ObjectGenerator')
        The name of the dummy with VREP child script for generating objects.
    vrep_function : str (default 'createObject_function')
        The name of the Lua function in child script that will use this config.

    Attributes:
    -----------
    configInts : list of ints
        Integers passed to Lua function in VREP
    configFloats : list of floats
        Floats pass to Lua function in VREP
    configStrings : list of strings
        Strings to pass to Lua function in VREP
    configBuffer : bytearray
        Buffer to pass to Lua function in VREP (unused)
    """
    colors = {'red': ColorParam(rgb=[1, 0, 0], mass=[0.1]),
              'green': ColorParam(rgb=[0, 1, 0], mass=[0.2]),
              'blue': ColorParam(rgb=[0, 0, 1], mass=[0.3]),
              'black': ColorParam(rgb=[0, 0, 0], mass=[0.4]),
              'white': ColorParam(rgb=[1, 1, 1], mass=[0.5]),
              'yellow': ColorParam(rgb=[1, 1, 0], mass=[0.6]),
              'orange': ColorParam(rgb=[1, 0.5, 0], mass=[0.7]),
              'grey': ColorParam(rgb=[0.5, 0.5, 0.5], mass=[0.8]),
              'purple': ColorParam(rgb=[0.5, 0, 1], mass=[0.9]),
              'teal': ColorParam(rgb=[0, 1, 1], mass=[1.0])}

    kinds = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'cone': 3}

    # can set these defaults to constrain random group configurations
    default_obj_kind = 'cylinder'
    default_prim_size = [0.2, 0.2, 0.2]
    default_conn_size = [0.05, 0.05, 0.6]
    default_offset = [0.0, 0.0, 0.3]

    def __init__(self, name, kind, color, xyz, size, 
                 vrep_dummy='ObjectGenerator',
                 vrep_function='createObject_function'):

        self.name = name
        self.kind = self.kinds[kind]  # convert string to VREP integer code
        self.xyz = xyz
        self.size = size
        self.rgb = self.colors[color].rgb
        self.mass = self.colors[color].mass

        self.vrep_dummy = vrep_dummy
        self.vrep_function = vrep_function

    @property
    def inputInts(self):
        '''List of ints passed to Lua function in VREP'''
        return [self.kind]

    @property
    def inputFloats(self):
        '''List of floats passed to Lua function in VREP'''
        return self.size + self.rgb + self.xyz + self.mass

    @property
    def inputStrings(self):
        '''List of strings passed to Lua function in VREP'''
        return [self.name]

    @property
    def inputBuffer(self):
        '''Buffer passed to Lua function in VREP (required by API)'''
        return bytearray()



class GroupConfig(object):
    """Collects object configs and automatically computes position offsets so
    that the objects are appropriately spaced once built and grouped together.

    Parameters:
    ----------

    xyz: list of floats
        The coordinates for the base object in the group.
    vrep_dummy : str (default 'ObjectGenerator')
        The name of the dummy with VREP child script for grouping objects.
    vrep_function : str (default 'createGroup_function')
        The name of the Lua function in child script that will use this config.

    """
    def __init__(self, xyz, 
                 vrep_dummy='ObjectGenerator', 
                 vrep_function='groupObjects_function'):

        self.xyz = xyz

        self.primitives = []
        self.connectors = []

        self.vrep_dummy = vrep_dummy
        self.vrep_function = vrep_function

    @property
    def all_object_configs(self):
        '''Return all of the object configs associated with this group'''
        return self.primitives + self.connectors

    def add_primitive_config(self, config_object):
        '''Add config for a primitive object after updating coordinates'''        
        scale = 2 * len(self.primitives)  # space as (prim, conn, prim...)
        offset = [c * scale for c in ObjectConfig.default_offset]
        xyz = [c1 + c2 for c1, c2 in zip(self.xyz, offset)]       
        
        config_object.xyz = xyz

        self.primitives.append(config_object)

    def add_connector_config(self, config_object):
        '''Add config for a connector object after updating coordinates'''
        scale = 2 * len(self.connectors) + 1  # space as (prim, conn, prim...)
        offset = [c * scale for c in ObjectConfig.default_offset]
        xyz = [c1 + c2 for c1, c2 in zip(self.xyz, offset)]

        config_object.xyz = xyz

        self.connectors.append(config_object)

    def random_configs(self, max_primitives=1):
        '''Randomly create primitive and connector object configs up to max'''
        max_connectors = max(max_primitives-1, 1)

        # randomly create each primitive object config
        for n in range(max_primitives):
            name = 'Primitive' + str(len(self.primitives))
            color = np.random.choice(list(ObjectConfig.colors.keys()))

            config = ObjectConfig(name=name,
                                  kind=ObjectConfig.default_obj_kind,
                                  size=ObjectConfig.default_prim_size,
                                  color=color,
                                  xyz=self.xyz)

            self.add_primitive_config(config)


        # randomly create each connection object config
        for n in range(max_connectors):
            name = 'Connector' + str(len(self.connectors))
            color = np.random.choice(list(ObjectConfig.colors.keys()))

            config = ObjectConfig(name=name,
                                  kind=ObjectConfig.default_obj_kind,
                                  size=ObjectConfig.default_conn_size,
                                  color=color,
                                  xyz=self.xyz)

            self.add_connector_config(config)