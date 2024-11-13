
def identity(hand, hist, pos, mean):
    """No abstraction"""
    return hand, hist


def simple(hand, hist, pos, mean):
    """Abstract away the entire history"""
    new_hist = ()
    return hand, new_hist


def simple_hand(hand, hist, pos, mean):
    """Abstract away the entire hand"""
    return [], hist


def naive(hand, hist, pos, mean):
    """Least possible states"""
    return [], ()


def possible(hand, hist, pos, mean):
    """For non-betting actions only use possible cards"""
    if 'Bet' in pos or 'Call' in pos:
        new_hand = hand
    else:
        new_hand = pos
    return new_hand, hist


def advanced(hand, hist, pos, mean):
    """Advanced abstraction which replaces the hand with a list containing avg hand, number of suits and
    number of high cards, and the history without betting order"""

    if len(hand) == 0:
        hand_str = 0
    else:
        hand_str = sum(map(lambda x: x[1], hand)) / len(hand)

    new_hist = (hist.count('Call'),)
    suits = set()
    for card in hand:
        suits.add(card[0])
    high_cards = len([x for x in hand if x[1] > mean])
    new_hand = [hand_str, len(suits), high_cards]

    return new_hand, new_hist


def adv_hand(hand, hist, pos, mean):
    """Abstract away the hand to a list containing avg hand, number of suits and number of high cards,"""

    if len(hand) == 0:
        hand_str = 0
    else:
        hand_str = sum(map(lambda x: x[1], hand)) / len(hand)

    suits = set()
    for card in hand:
        suits.add(card[0])
    high_cards = len([x for x in hand if x[1] > mean])
    new_hand = [hand_str, len(suits), high_cards]

    return new_hand, hist



