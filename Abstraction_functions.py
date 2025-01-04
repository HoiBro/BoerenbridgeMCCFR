
def identity(hand, trump, hist, wins, pos, mean):
    """No abstraction"""
    return hand, trump, hist, wins


def naive(hand, trump, hist, wins, pos, mean):
    """Least possible states"""
    return [], (), (), [0, 0]


def simple(hand, trump, hist, wins, pos, mean):
    """Abstract away the entire history"""
    return hand, trump, (), wins


def simple_hand(hand, trump, hist, wins, pos, mean):
    """Abstract away the entire hand"""
    return [], trump, hist, wins


def bets(hand, trump, hist, wins, pos, mean):
    """No history except for the bets and amount of wins"""
    new_hist = hist[:2]
    return hand, trump, new_hist, wins


def suit(hand, trump, hist, wins, pos, mean):
    """Abstract away the trumps rank"""
    return hand, trump[0], hist, wins


def suitbet(hand, trump, hist, wins, pos, mean):
    """Combination of "bets" and "suit" abstractions"""
    new_hist = hist[:2]
    return hand, trump[0], new_hist, wins


def advanced(hand, trump, hist, wins, pos, mean):
    """Abstract away the hand to a list containing avg hand, number of suits and number of high cards"""
    if len(hand) == 0:
        hand_str = 0
    else:
        hand_str = sum(map(lambda x: x[1], hand)) / len(hand)
    
    suits = set()
    for card in hand:
        suits.add(card[0])
    high_cards = len([x for x in hand if x[1] > mean])
    new_hand = [hand_str, len(suits), high_cards]

    return new_hand, trump, hist, wins
