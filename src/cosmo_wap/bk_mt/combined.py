from cosmo_wap.lib.utils import add_empty_methods_bk

# Empty multi-tracer stubs - placeholders until the multi-tracer terms are implemented.
# Each class is decorated so every multipole returns a zero array (see add_empty_methods_bk).


############################################################### wide separation terms
@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WA1:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WA2:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WARR:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class RR1:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class RR2:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WAGR:
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class RRGR:
    pass


############################################################### composite terms
@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WS:
    """Pure wide separation terms, RR + WA"""
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class WSGR:
    """Full wide separation effects including relativistic mixing"""
    pass


@add_empty_methods_bk('l0', 'l1', 'l2', 'l3')
class Full:
    """WS + GR"""
    pass
