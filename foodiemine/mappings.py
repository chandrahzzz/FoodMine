"""Hand-built concept hierarchies used by the warehouse and the miners.

location -> area -> zone -> city   (geography hierarchy for roll-up / drill-down)
cuisine  -> cuisine group           (cuisine hierarchy)

Zones are an approximate split of Bangalore into five parts based on where
each neighbourhood lies relative to the city centre (MG Road).
"""
import re

ZONES = {
    "Central": [
        "Brigade Road", "Central Bangalore", "Church Street", "City Market",
        "Commercial Street", "Cunningham Road", "Infantry Road", "Langford Town",
        "Lavelle Road", "MG Road", "Majestic", "Race Course Road", "Residency Road",
        "Richmond Road", "Sankey Road", "Seshadripuram", "Shanti Nagar",
        "Shivajinagar", "St. Marks Road", "Ulsoor", "Vasanth Nagar", "Wilson Garden",
    ],
    "North": [
        "Frazer Town", "HBR Layout", "Hebbal", "Hennur", "Jakkur", "Jalahalli",
        "Kalyan Nagar", "Kammanahalli", "Malleshwaram", "Nagawara", "New BEL Road",
        "North Bangalore", "Peenya", "RT Nagar", "Sadashiv Nagar", "Sahakara Nagar",
        "Sanjay Nagar", "Yelahanka", "Yeshwantpur",
    ],
    "East": [
        "Banaswadi", "Bellandur", "Brookefield", "CV Raman Nagar", "Domlur",
        "East Bangalore", "ITPL Main Road, Whitefield", "Indiranagar",
        "Jeevan Bhima Nagar", "KR Puram", "Kaggadasapura", "Marathahalli",
        "Old Airport Road", "Old Madras Road", "Rammurthy Nagar", "Sarjapur Road",
        "Thippasandra", "Varthur Main Road, Whitefield", "Whitefield",
    ],
    "South": [
        "BTM", "Banashankari", "Bannerghatta Road", "Basavanagudi", "Bommanahalli",
        "Ejipura", "Electronic City", "HSR", "Hosur Road", "JP Nagar", "Jayanagar",
        "Kanakapura Road", "Koramangala", "Koramangala 1st Block",
        "Koramangala 2nd Block", "Koramangala 3rd Block", "Koramangala 4th Block",
        "Koramangala 5th Block", "Koramangala 6th Block", "Koramangala 7th Block",
        "Koramangala 8th Block", "Kumaraswamy Layout", "South Bangalore", "Uttarahalli",
    ],
    "West": [
        "Basaveshwara Nagar", "Kengeri", "Magadi Road", "Mysore Road", "Nagarbhavi",
        "Rajajinagar", "Rajarajeshwari Nagar", "Vijay Nagar", "West Bangalore",
    ],
}
LOCATION_TO_ZONE = {loc: zone for zone, locs in ZONES.items() for loc in locs}


def location_to_area(location: str) -> str:
    """Collapse sub-localities into one area, e.g. 'Koramangala 5th Block' -> 'Koramangala'."""
    area = re.sub(r"\s+\d+(st|nd|rd|th) Block$", "", location)
    if area.endswith(", Whitefield"):
        area = "Whitefield"
    return area


CUISINE_GROUPS = {
    "North Indian": [
        "North Indian", "Mughlai", "Lucknowi", "Awadhi", "Kebab", "Modern Indian",
        "Indian",
    ],
    "South Indian": [
        "South Indian", "Andhra", "Kerala", "Chettinad", "Mangalorean", "Hyderabadi",
        "Tamil", "Biryani",
    ],
    "Regional Indian": [
        "Bengali", "Oriya", "Assamese", "Naga", "North Eastern", "Gujarati",
        "Maharashtrian", "Rajasthani", "Goan", "Konkan", "Malwani", "Parsi", "Bohri",
        "Sindhi", "Bihari", "Kashmiri",
    ],
    "Asian": [
        "Chinese", "Asian", "Thai", "Japanese", "Tibetan", "Momos", "Korean",
        "Vietnamese", "Malaysian", "Indonesian", "Burmese", "Singaporean", "Sushi",
        "Cantonese", "Mongolian", "Pan Asian", "Nepalese", "Sri Lankan",
    ],
    "Continental & Western": [
        "Continental", "Italian", "American", "European", "Mediterranean", "Mexican",
        "French", "Spanish", "Greek", "Tex-Mex", "British", "German", "Portuguese",
        "Russian", "Jewish", "Australian", "Belgian", "South American", "African",
        "Steak", "BBQ", "Grill",
    ],
    "Middle Eastern": [
        "Arabian", "Lebanese", "Middle Eastern", "Turkish", "Afghan", "Afghani",
        "Iranian",
    ],
    "Fast Food & Snacks": [
        "Fast Food", "Street Food", "Pizza", "Burger", "Rolls", "Finger Food",
        "Sandwich", "Wraps", "Hot dogs", "Roast Chicken", "Charcoal Chicken", "Bar Food",
    ],
    "Desserts & Bakery": ["Desserts", "Bakery", "Ice Cream", "Mithai", "Paan"],
    "Beverages & Cafe": [
        "Beverages", "Cafe", "Juices", "Tea", "Coffee", "Bubble Tea", "Drinks Only",
    ],
    "Healthy": ["Salad", "Healthy Food", "Vegan"],
    "Seafood": ["Seafood", "Raw Meats"],
}
CUISINE_TO_GROUP = {c: g for g, cs in CUISINE_GROUPS.items() for c in cs}


def cuisine_to_group(cuisine: str) -> str:
    return CUISINE_TO_GROUP.get(cuisine, "Other")
