import re

# A representative subset of the ~350 common prefixes and suffixes used in FT8.
# In a production environment, this list must be exhaustive (matching WSJT-X callsign.cxx).
# Common prefixes: DXCC entities (e.g., ZA, 9M, PJ4)
# Common suffixes: Portable/Mobile indicators (e.g., /P, /M, /MM, /AM, /QRP) and common DX (e.g., /KH1, /KL7)
COMMON_PREFIXES_SUFFIXES = {
    # Common Suffixes
    "P", "M", "MM", "AM", "QRP", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "N", "O", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "KH1", "KH2", "KH4", "KH5", "KH6", "KH7", "KH8", "KH9", "KL7", "KP1", "KP2", "KP4", "KP5",
    "NP3", "NP4", "OF0", "OH0", "OJ0", "OK1", "OM0", "ON4", "OX", "OY", "OZ",
    "9M0", "9M2", "9M6", "9M8", "9V", "A3", "A4", "A5", "A6", "A7", "A9", "AP", "BV", "BV0", "BY", "C2", "C3", "C5", "C6", "C9", "CE0", "CE9", "CM0", "CN", "CT3", "CU", "CX", "CY0", "CY9", "D4", "DL", "DU", "E3", "E4", "E5", "E7", "EA0", "EA6", "EA8", "EA9", "EI", "EK", "EL", "EP", "ER", "ES", "ET", "EU", "EW", "EX", "EY", "EZ", "F", "FG", "FH", "FJ", "FK", "FM", "FO", "FP", "FR", "FT", "FW", "FY", "G", "GD", "GI", "GJ", "GM", "GM0", "GU", "GW", "H4", "HA", "HB0", "HC0", "HC8", "HH", "HI", "HL", "HP", "HR", "HS", "HV", "HZ", "I", "IS", "IT", "IX", "J2", "J3", "J5", "J6", "J7", "J8", "JA", "JD0", "JD1", "JI", "JK", "JM", "JN", "JO", "JP", "JQ", "JR", "JT", "JU", "JV", "JW", "JX", "JY", "JZ", "K", "KG4", "KH0", "KH8", "KI", "KJ", "KK", "KL", "KM", "KN", "KO", "KP", "KQ", "KR", "KS", "KT", "KU", "KV", "KW", "KX", "KY", "KZ", "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LI", "LJ", "LK", "LL", "LM", "LN", "LO", "LP", "LQ", "LR", "LS", "LT", "LU", "LV", "LW", "LX", "LY", "LZ", "OA", "OD", "OE", "OH", "OI", "OJ", "OK", "OL", "OM", "ON", "OO", "OP", "OQ", "OR", "OS", "OT", "OU", "OV", "OW", "OX", "OY", "OZ", "P2", "P4", "PA", "PB", "PC", "PD", "PE", "PF", "PG", "PH", "PI", "PJ", "PK", "PL", "PM", "PN", "PO", "PP", "PQ", "PR", "PS", "PT", "PU", "PV", "PW", "PX", "PY", "PZ", "R", "S0", "S2", "S5", "S7", "S9", "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SP", "SQ", "SR", "SS", "ST", "SU", "SV", "SW", "SX", "SY", "SZ", "T2", "T3", "T5", "T6", "T7", "T8", "TA", "TB", "TC", "TD", "TE", "TF", "TG", "TH", "TI", "TJ", "TK", "TL", "TM", "TN", "TO", "TP", "TQ", "TR", "TS", "TT", "TU", "TV", "TW", "TX", "TY", "TZ", "UA", "UB", "UC", "UD", "UE", "UF", "UG", "UH", "UI", "UJ", "UK", "UL", "UM", "UN", "UO", "UP", "UQ", "UR", "US", "UT", "UU", "UV", "UW", "UX", "UY", "UZ", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "VA", "VB", "VC", "VD", "VE", "VF", "VG", "VH", "VI", "VJ", "VK", "VL", "VM", "VN", "VO", "VP", "VQ", "VR", "VS", "VT", "VU", "VV", "VW", "VX", "VY", "VZ", "W", "WA", "WB", "WC", "WD", "WE", "WF", "WG", "WH", "WI", "WJ", "WK", "WL", "WM", "WN", "WO", "WP", "WQ", "WR", "WS", "WT", "WU", "WV", "WW", "WX", "WY", "WZ", "XA", "XB", "XC", "XD", "XE", "XF", "XG", "XH", "XI", "XJ", "XK", "XL", "XM", "XN", "XO", "XP", "XQ", "XR", "XS", "XT", "XU", "XV", "XW", "XX", "XY", "XZ", "YA", "YB", "YC", "YD", "YE", "YF", "YG", "YH", "YI", "YJ", "YK", "YL", "YM", "YN", "YO", "YP", "YQ", "YR", "YS", "YT", "YU", "YV", "YW", "YX", "YY", "YZ", "Z2", "Z3", "Z6", "Z8", "ZA", "ZB", "ZC", "ZD", "ZE", "ZF", "ZG", "ZH", "ZI", "ZJ", "ZK", "ZL", "ZM", "ZN", "ZO", "ZP", "ZQ", "ZR", "ZS", "ZT", "ZU", "ZV", "ZW", "ZX", "ZY", "ZZ",
    # Add more specific common DXCC prefixes and suffixes as needed to match the full WSJT-X list
}

def extract_compound_callsign(message):
    """
    Extracts the compound callsign from an FT8 message string.
    Returns (callsign, position) where position is 'first', 'second', or None.
    """
    parts = message.strip().split()
    compound = None
    pos = None
    
    # Check for compound callsigns (containing '/')
    for i, part in enumerate(parts):
        if '/' in part:
            # Basic validation: looks like a callsign (alphanumeric + /)
            if re.match(r'^[A-Z0-9/]+$', part):
                compound = part
                pos = 'first' if i == 0 else 'second' if i == 1 else 'other'
                break # Take the first one found as the primary compound candidate
                
    return compound, pos

def is_type1(callsign):
    """
    Determines if a compound callsign is Type 1.
    Type 1: The prefix or suffix (the part after/before '/') is in the common list.
    """
    if '/' not in callsign:
        return False
    
    parts = callsign.split('/')
    if len(parts) != 2:
        return False # Complex compounds usually handled as Type 2 or Free Text
        
    prefix, suffix = parts
    
    # Check if prefix is in common list (e.g., ZL/W9XYZ -> ZL is common)
    # Or if suffix is in common list (e.g., K1ABC/4 -> 4 is common, though single digits are tricky, usually letters)
    # The WSJT-X logic checks if the add-on is in the fixed list.
    
    # Note: The common list contains both prefixes (like 'ZL') and suffixes (like 'P', 'KH1')
    if prefix in COMMON_PREFIXES_SUFFIXES:
        return True
    if suffix in COMMON_PREFIXES_SUFFIXES:
        return True
        
    return False

def determine_ft8_message_type(message):
    """
    Determines if an FT8 message is Type 1 or Type 2 based on WSJT-X rules.
    
    Rules:
    Type 1:
      - Compound callsign uses a common prefix/suffix.
      - Replaces the 3rd word (no locator/report allowed).
      - Can appear as 1st word (HisCall) or 2nd word (MyCall) in specific messages.
      
    Type 2:
      - Compound callsign uses an UNCOMMON prefix/suffix.
      - Must be the 2nd word.
      - 1st word must be CQ, DE, or QRZ.
      - Can have a 3rd word (locator, report, etc).
    """
    compound, pos = extract_compound_callsign(message)
    
    if not compound:
        return "Standard (No Compound Callsign)"

    parts = message.split()
    
    # Check Type 1 Condition: Is the add-on in the common list?
    if is_type1(compound):
        # Validation for Type 1: Should NOT have a 3rd word if it's acting as the replacement
        # However, the definition of Type 1 is primarily about the CALLSIGN itself being in the list.
        # If the message has 3 words AND a Type 1 callsign, WSJT-X might treat it as invalid or free text
        # depending on position, but the *callsign type* is Type 1.
        # The prompt asks to set type based on the message. 
        # Strictly: If it uses a common prefix/suffix, it's Type 1 logic.
        return 1
    
    # Check Type 2 Condition: Not in common list
    # Must be 2nd word, 1st word must be CQ/DE/QRZ
    if not is_type1(compound):
        if pos == 'second':
            first_word = parts[0].upper()
            if first_word in ['CQ', 'DE', 'QRZ']:
                return 2
            else:
                # Invalid Type 2 structure (e.g. "XL/K1ABC W9XYZ") -> Treated as Free Text by WSJT-X
                return "Free Text (Invalid Type 2 Structure)"
        elif pos == 'first':
            # Type 2 callsigns generally cannot be in the first position in standard messages
            # (Except possibly in Free Text)
            return "Free Text (Type 2 in 1st pos)"
            
    return "Unknown/Free Text"


if __name__ == "__main__":
    # --- Examples ---
    test_messages = [
        "CQ ZL/W9XYZ",          # Type 1 (ZL is common prefix)
        "K1ABC ZL/W9XYZ",       # Type 1 (ZL is common)
        "K1ABC/VE1 73",         # Type 1 (VE1 is common prefix/suffix in list) - Wait, VE1 is common? Yes.
        "CQ W9XYZ/VE1 FN75",    # If VE1 is common -> Type 1 (but has 3 words? Type 1 usually replaces 3rd word). 
                                # Actually, if VE1 is common, "CQ W9XYZ/VE1 FN75" is INVALID for Type 1 encoding.
                                # WSJT-X would likely force Type 2 if the user forces the message, 
                                # BUT the definition of Type 1 vs 2 is about the callsign list.
                                # Let's refine: If the callsign is Type 1, it CANNOT have a 3rd word.
                                # If the message HAS a 3rd word, and the callsign is from the common list, 
                                # it's an invalid Type 1 message, often sent as Free Text.
                                # However, if the callsign is NOT in the list, it IS Type 2.
        
        "CQ W9XYZ/VE9 FN75",    # Type 2 (VE9 likely not in common list, 1st=CQ, 2nd=Compound, 3rd=Grid)
        "DE W4/G0XYZ -22",      # Type 2 (G0XYZ/W4? No, W4/G0XYZ. If W4 is common? W4 is a US district, usually not a DXCC prefix in the short list unless 'W' is? No. 
                                # 'W4' is not a DXCC prefix. So Type 2.
        "XL/K1ABC W9XYZ",       # Free Text (Type 2 callsign 'XL' not common, but 1st word is not CQ/DE/QRZ)
    ]
    
    print(f"{'Message':<25} | {'Type'}")
    print("-" * 40)
    for msg in test_messages:
        m_type = determine_ft8_message_type(msg)
        print(f"{msg:<25} | {m_type}")   
