from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

def rgb2lab(r,g,b):
    return list(convert_color( 
        sRGBColor(r/255,g/255,b/255), LabColor).get_value_tuple())

def lab2rgb(l,a,b):
    return list(convert_color( 
        LabColor(l,a,b), sRGBColor).get_value_tuple())


def get_color_vis(l, description= None):
    """
    receives a RGB tuple and generate a html block with the
    associated color as background 
    to be used in jupyter for visualization
    """
    hex_rep  =  rgb2hex(l[0], l[1], l[2])
    
    if description is None:
        s =  "<div style='float:left;background:"+ hex_rep +";padding:10px;width:300px;'></div>"
    else:
        s =  "<table> \
        <td> " + description + "  </td> \
        <td><div style='float:left;background:"+ hex_rep +";padding:10px;width:300px;'></div></td> \
        </table>"

    return s

def get_incremental_color_vis(  tuples  ):
    
    s = "<table>"
    for n,c in tuples:
        hex_rep  =  rgb2hex(c[0], c[1], c[2])
        s = s + " <tr> "
        
        s = s +   "<td style='text-align:left;'> " + n + "  </td> <td><div style='float:left;background:"+ hex_rep +";padding:10px;width:300px;'></div></td>"
        
        
        s = s + "</tr>"
        
    s = s + "</table>"
    
    return s


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

