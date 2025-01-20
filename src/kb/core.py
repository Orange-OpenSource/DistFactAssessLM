# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from __future__ import annotations
import calendar
from contextlib import contextmanager
import functools
import itertools
from typing import Hashable, Iterable, Union
from collections.abc import Iterable as IterIsInst
import numpy as np
import re
from abc import ABC, ABCMeta, abstractmethod
from glob_core import Mongoable, MongoableEnum, SameType, TimeUnit

from utils.general import all_equal, create_switch_contextmanager
    
class Describable(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def __repr__(self) -> str:
        return str(self.label)
    

class TripleComp(MongoableEnum):
    SUBJECT = 0
    RELATION = 1
    OBJECT = 2

class Entity(Describable, Mongoable):
    def __init__(self, id : Hashable, label : str = None, description : str = None) -> None:
        super().__init__()
        self.id = id
        self._label = label
        self._description = description

    def __eq__(self, __value: object) -> bool:
        if type(self) is not type(__value):
            return False
        return self.id == __value.id

    def __hash__(self) -> int:
        return hash(self.id)
    
    @property
    def label(self) -> str:
        if self._label is not None:
            return self._label
        return self.id
    
    @property
    def description(self) -> str:
        return str(self._description)
    
    def __repr__(self) -> str:
        return self.label

    def _to_json(self) -> dict | list:
        return {
            'id' : self.id,
            'label' : self._label,
            'description' : self._description
        }

    @classmethod
    def _from_json(cls: type[SameType], d : dict | list) -> SameType:
        return cls(**d)

    def _identifier(self) -> Hashable:
        return self.id

    
    
class Relation(Entity):
    pass

class Literal(Entity):
    def __init__(self, id : Hashable) -> None:
        super().__init__(id)
    
    @property
    def description(self) -> str:
        return self.__class__.__name__
    
class Unit(Entity):
    def __init__(self, id: Hashable, symbol : str, label: str = None, description: str = None) -> None:
        super().__init__(id, label, description)
        # assert symbol is not None, "symbol must be of type str"
        self._symbol = symbol

    @property
    def symbol(self) -> str:
        return self._symbol
    
    def _to_json(self) -> dict | list:
        d = super()._to_json()
        d['symbol'] = self._symbol

class String(Literal):
    def __init__(self, txt : str) -> None:
        super().__init__(txt)

    def __repr__(self) -> str:
        return '"%s"' % super().__repr__()

class DateFormatIncorrect(Exception):
    def __init__(self, txt : str) -> None:
        self.txt = txt
        super().__init__('Accepted date formats are : YYYY, YYYY-MM, YYYY-MM-DD with possibly a "-" sign at the start. Found "%s"' % self.txt)

@functools.total_ordering
class Date(Literal):
    # _FROM_STRING_PAT = re.compile(r'^-?([0-9]+?|[0-9]+?-[0-9]{2}|[0-9]+?-[0-9]{2}-[0-9]{2})$')
    _FROM_STRING_PAT = re.compile(r'^[-\+]?([0-9]+?)(-[0-9]{2})?(-[0-9]{2})?$')
    _LEVEL_PAT = re.compile(r'^<M8\[(\w+?)\]$')
    CHAR2LEVEL = {
        'D' : TimeUnit.DAY,
        'M' : TimeUnit.MONTH,
        'Y' : TimeUnit.YEAR
    }
    LEVEL2CHAR = {v:k for k,v in CHAR2LEVEL.items()}
    _level_heterogeneous_comparison_context = False

    @staticmethod
    def from_string(txt : str) -> Date:
        """Initialize Date from string. 

        Args:
            txt (str): Date in text description. Accepted format : YYYY-MM-DD, YYYY-MM, YYYY with possibly a "-" or "+" sign at the start.
            Note : YYYY can be as long as necessary

        Raises:
            DateFormatIncorrect: When date format is not supported

        Returns:
            Date: Date object
        """
        txt = txt.lstrip('+')
        m = Date._FROM_STRING_PAT.match(txt)
        if not m:
            raise DateFormatIncorrect(txt)
        if m.group(3):
            dt = np.datetime64(txt, 'D')
        elif m.group(2):
            dt = np.datetime64(txt, 'M')
        elif m.group(1):
            dt = np.datetime64(txt, 'Y')
        else:
            raise DateFormatIncorrect(txt)
        return Date(dt)
    
    def __init__(self, date : np.datetime64) -> None:
        if not isinstance(date, np.datetime64):
            raise TypeError("'date' must be a scalar numpy datetime64 object")
        super().__init__(date)
        self.id : np.datetime64
        self._diff_level_comp = False

    @property
    def level_heterogeneous_comparison_enabled(self) -> bool:
        if Date._level_heterogeneous_comparison_context:
            return True
        return self._diff_level_comp

    @level_heterogeneous_comparison_enabled.setter
    def level_heterogeneous_comparison_enabled(self, value : bool):
        assert isinstance(value, bool)
        self._diff_level_comp = value

    @property
    def label(self) -> str:
        return np.datetime_as_string(self.id, unit=Date.LEVEL2CHAR[self.level])

    @property
    def level(self) -> TimeUnit:
        c = re.match(Date._LEVEL_PAT, self.id.dtype.str).group(1)
        return Date.CHAR2LEVEL[c]
    
    @property
    def day(self) -> int:
        if self.level > TimeUnit.DAY:
            return None
    
        day = (self.id - self.id.astype('datetime64[M]')).astype(int) + 1
        return int(day)

    @property
    def month(self) -> int:
        if self.level > TimeUnit.MONTH:
            return None
    
        month = self.id.astype('datetime64[M]').astype(int) % 12 + 1
        return int(month)
    
    @property
    def year(self) -> int:
        if self.level > TimeUnit.YEAR:
            return None
        year = self.id.astype('datetime64[Y]').astype(int) + 1970
        return int(year)
    
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, Date):
            raise TypeError("Comparison not supported between %s and %s" % (type(self).__name__, type(other).__name__))
        if not self.level_heterogeneous_comparison_enabled and other.level != self.level:
            raise ValueError("Comparison not supported between dates of different level : self.level = %s, other.level = %s" % (self.level.name, other.level.name))
        elif self.level_heterogeneous_comparison_enabled:
            self, other = self.level_completion('max'), other.level_completion('min')
        return self.id < other.id
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Date):
            return False
        if not isinstance(__value.id, np.datetime64):
            return False
        return self.id == __value.id
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    def level_completion(self, type : str) -> Date:
        """Complete an incomplete date. Example : 2021-05 --> 2021-05-01 (when type='min') or 2021-05-31 (when type='max')

        Args:
            type (str): Type of completion. It takes one of these two values : 'max' (complete to the highest possible date) 
            or 'min' (complete to the lowest possible date).  

        Returns:
            Date: Completed date
        """
        if type == 'mean':
            min, max = self.level_completion('min'), self.level_completion('max')
            return Interval(min, max).midpoint()
        if self.level == TimeUnit.DAY:
            return self
        elif self.level == TimeUnit.MONTH:
            if type == 'max':
                n_days = calendar.monthrange(self.year, self.month)[1]
                return Date.from_string("%s-%s" % (self, n_days))
            else:
                return Date.from_string("%s-01" % self)
        elif self.level == TimeUnit.YEAR:
            if type == 'max':
                return Date.from_string("%s-12-31" % self)
            else:
                return Date.from_string("%s-01-01" % self)
        raise Exception("This should be impossible! What is the value of self.level?!")
    
    def change_level(self, level : TimeUnit) -> Date:
        year, month, day = self.year, self.month, self.day
        if level == TimeUnit.DAY:
            month = 1 if month is None else month
            day = 1 if day is None else day
        elif level == TimeUnit.MONTH:
            month = 1 if month is None else month
        return Date.from_string("%s-%02d-%02d" % (year,month, day))
    
    def _to_json(self) -> dict | list:
        d = {'date' : self.label}
        return d
    
    @classmethod
    def _from_json(cls: type[SameType], d : dict | list) -> SameType:
        d['date'] = np.datetime64(d['date'])
        return cls(**d)
    
class InfDate(Date):
    def __init__(self, minus:bool) -> None:
        self.minus = minus
        super().__init__(str(self))
    
    def __repr__(self) -> str:
        symb = "-" if self.minus else '+'
        return "%sInfDate" % symb
    
    @property
    def day(self) -> int:
        raise Exception('InfDate has no day')
    
    @property
    def month(self) -> int:
        raise Exception('InfDate has no month')
    
    @property
    def year(self) -> int:
        raise Exception('InfDate has no year')
    
    @property
    def level(self) -> int:
        raise Exception('InfDate has no level')
    
    @property
    def label(self) -> str:
        return str(self)
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, Date):
            raise TypeError("Comparison not supported between %s and %s" % (type(self).__name__, type(other).__name__))
        if type(other) is InfDate:
            return self.minus > other.minus 
        return self.minus
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Date):
            return False
        if type(__value) is InfDate:
            return self.minus == __value.minus
        return False
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    def _to_json(self) -> dict | list:
        d = {"minus" : self.minus}
        return d
    
    def _identifier(self) -> Hashable:
        return self.minus


enable_level_heterogeneous_comparison = create_switch_contextmanager(Date, "level_heterogeneous_comparison_enabled")


@functools.total_ordering
class Quantity(Literal):
    def __init__(self, value : Union[float, int], unit : Unit = None) -> None:
        super().__init__((value, unit))
    
    def __lt__(self, other : Quantity):
        if not isinstance(other, Quantity):
            raise TypeError("Comparison not supported between '%s' and '%s'" % (type(self), type(other)))
        if self.unit != other.unit:
            raise ValueError("Comparison not supported between quantities of different units")
        return self.value < other.value
    
    @property
    def unit(self) -> Unit:
        return self.id[1]
    
    @property
    def value(self) -> Union[int, float]:
        return self.id[0]
    
    @property
    def label(self) -> str:
        s = "%s" % self.value
        if self.unit is not None:
            symb = self.unit.symbol
            if symb is None:
                symb = "UnkUnit"
            s += " %s" % symb
        return s
    
    def _to_json(self) -> dict | list:
        d = {
            'value' : self.value,
            'unit' : self.unit
        }
        return d

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(**d)

class Interval(Mongoable):
    def __init__(self, start : Date, end : Date) -> None:
        super().__init__()
        with enable_level_heterogeneous_comparison():
            if start is not None and end is not None and start > end:
                raise ValueError("Must : start <= end! Found: start=%s, end=%s" % (start, end))
        self.start = start
        self.end = end
    
    def is_point(self):
        return self.start == self.end and self.start is not None
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Interval):
            return (self.start, self.end) == (value.start, value.end)
        return False
        
    def is_included_in(self, other : Interval) -> bool:
        # NOTE: More complicated than this if dates are of different precisions
        try:
            return self.start >= other.start and self.end <= other.end
        except TypeError:
            return False
        
    def midpoint(self) -> Date:
        if self.start is None or self.end is None:
            return None
        st = self.start.level_completion('min')
        et = self.end.level_completion('max')
        return Date(st.id + (et.id - st.id) / 2)
    
    def __repr__(self) -> str:
        if self.is_point():
            return "%s (point)" % self.start
        elif self.start is None and self.end is None:
            return "unspecified"
        return "%s to %s" % (self.start, self.end)
    
    def _to_json(self) -> dict | list:
        d = {
            'start' : self.start,
            'end' : self.end
        }
        return d
    
    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(**d)
    
    def _identifier(self) -> functools.Any:
        return self._to_json()

    def intersection(self, interval : Interval) -> Interval:
        return Interval.intersection_many([self, interval])

    @staticmethod
    def intersection_many(intervals : list[Interval]) -> Interval:

        # Sort intervals based on the start time
        intervals.sort(key=lambda x: x.start)

        # Start with the first interval as the initial intersection
        current_intersection = intervals[0]

        # Iterate through the intervals and update the intersection
        for interval in intervals[1:]:
            # Update the start of the intersection to the max of current starts
            current_start = max(current_intersection.start, interval.start)
            # Update the end of the intersection to the min of current ends
            current_end = min(current_intersection.end, interval.end)

            # If the current start is after the current end, there is no intersection
            if current_start > current_end:
                return None  # No common intersection

            # Update the current intersection
            current_intersection = Interval(current_start, current_end)

        return current_intersection


class EntityContainer:
    """Describes a class of objects that contain entities 
    """
    def iter_comp(self) -> Iterable[Entity]:
        """Iterate through all entities present in this object

        Returns:
            Iterable[Entity]
        """
        raise NotImplementedError
    
    @staticmethod
    def iter_comp_extended(iterable : list | dict | tuple | EntityContainer | Entity) -> Iterable[Entity]:
        """Iterate through all entities present in the given object
        """
        if isinstance(iterable, Entity):
            yield iterable
        elif isinstance(iterable, EntityContainer):
            yield from iterable.iter_comp()
        elif isinstance(iterable, (IterIsInst, dict)):
            if isinstance(iterable, dict):
                iterable = itertools.chain(iterable.keys(), iterable.values())
            for x in iterable:
                yield from EntityContainer.iter_comp_extended(x)

class Proposition(ABC):
    pass


class Triple(Proposition, Mongoable, EntityContainer):
    def __init__(self, subject : Entity, relation : Entity, object : Entity) -> None:
        super().__init__()
        self.subject = subject
        self.relation = relation
        self.object = object
    
    def __repr__(self) -> str:
        return "(%s, %s, %s)" % (self.subject, self.relation, self.object)
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Triple):
            return False
        return self.subject == value.subject and self.relation == value.relation and self.object == value.object
    
    def __hash__(self) -> int:
        return hash(self.subject) + hash(self.relation) + hash(self.object)
    
    def _to_json(self) -> dict | list:
        d = {
            'subject' : self.subject,
            'relation': self.relation,
            'object' : self.object
        }
        return d

    def _identifier(self) -> Hashable:
        return self.subject._identifier(), self.relation._identifier(), self.object._identifier()

    def get_comp(self, comp : TripleComp) -> Entity:
        if comp == TripleComp.SUBJECT:
            return self.subject
        elif comp == TripleComp.OBJECT:
            return self.object
        elif comp == TripleComp.RELATION:
            return self.relation
    
    def to_sro_str(self) -> tuple[str]:
        return [x.label for x in self.to_sro()]
        
    def to_sro(self) -> tuple[Entity]:
        return self.subject, self.relation, self.object
    
    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(**d)
    
    def iter_comp(self) -> Iterable[Entity]:
        return self.to_sro()

class TimedTriple(Triple):
    def __init__(self, subject: Entity, relation: Entity, object: Entity, valid_between : Interval) -> None:
        super().__init__(subject, relation, object)
        self.valid_between = valid_between
    
    def __repr__(self) -> str:
        if self.valid_between is not None:
            start, end = self.valid_between.start, self.valid_between.end
        else:
            start, end = None, None
        return "(%s, %s, %s, start=%s, end=%s)" % (self.subject, self.relation, self.object, start, end)
    
    def _to_json(self) -> dict | list:
        d = Triple._to_json(self)
        d['valid_between'] = self.valid_between
        return d

    def _identifier(self) -> Hashable:
        return Triple._identifier(self) + (self.valid_between._identifier(),)
    

class Group(EntityContainer):
    def from_triples(self, triples : Iterable[Triple]) -> Group:
        assert all_equal([t.subject for t in triples]), "All subjects in triples must be equal"
        assert all_equal([t.relation for t in triples]), "All relations in triples must be equal"
        return Group.__init__(triples[0].subject, triples[0].relation, [t.object for t in triples])
    
    def __init__(self, subject : Entity, relation : Relation, objects : Iterable[Entity]) -> None:
        self.subject = subject
        self.relation = relation
        self.objects = objects

    def iter_comp(self) -> Iterable[Entity]:
        yield self.subject
        yield self.relation
        for x in self.objects:
            yield x

class Query(ABC):
    pass

class TripleQuery(Query):
    def __init__(self, subject : Entity | Iterable[Entity] = None, 
                 relation : Relation | Iterable[Relation] = None, object : Entity | Iterable[Relation] = None) -> None:
        """Query meaning : Find all triples that contain the given subject, relation, and object (if provided). If one of these entities is not given, this function ignores it for the filtering process.

        When retrieving, only exact match is considered a hit.

        If a list is provided for subject for instance, it means "Find all the triple with one of the mentioned subject" 
        (there is an OR operator between subjects)
        

        Args:
            subject (Entity | Iterable[Entity]): Subject(s) to look for
            relation (Relation | Iterable[Relation]) : Relation(s) to look for
            object (Entity | Iterable[Entity]) : Object(s) to look for 
        """
        super().__init__()
        # assert sum(isinstance(x, abc.Iterable) for x in (subject, relation, object)) <= 1, "Many iterable triple queries are not supported!"
        self.subject = subject
        self.relation = relation
        self.object = object

    @staticmethod
    def from_triple(triple : Triple | list[Triple]) -> TripleQuery:
        """Initialize query from triple(s). If many triples (s_i, r_i, o_i) are provided,
        they are combined as follows : TripleQuery(subject=(s_1,...,s_n), relation=(r_1,...,r_n), object=(o_1,...,o_n))

        Args:
            triple (Triple): Triple or list of Triples

        Returns:
            TripleQuery: Query
        """
        if isinstance(triple, Triple):
            triple = [triple]
        subject, relation, object = list(zip(*[tr.to_sro() for tr in triple]))
        return TripleQuery(subject, relation, object)
    
    def __repr__(self) -> str:
        triple = tuple(("?" if x is None else str(x)) for x in (self.subject,self.relation,self.object))
        # entity_no_label = [type(x) in [Entity, Relation] and x._label is None for x in (self.subject,self.relation,self.object)]
        # triple = tuple(x[1:-1] if (isinstance(x, (Quantity, Date)) or no_label) else x for x, no_label in zip(triple, entity_no_label))
        return 'Query(subject=%s, relation=%s, object=%s)' % triple

class TimedTripleQuery(TripleQuery):
    def __init__(self, subject : Entity | Iterable[Entity] = None, 
                 relation : Relation | Iterable[Relation] = None, object : Entity | Iterable[Entity] = None,
                 valid_at : Interval | Date | str | Iterable[Interval | Date | str] = None) -> None:
        """Query meaning : Find all timed triples that contain the given subject, relation, object, and validity period (if provided). 
        If one of these entities is not given, this function ignores it for the filtering process.

        When retrieving, only exact match is considered a hit for subject, relation, and object. For valid_at, see how it works below.

        IMPORTANT: It you want to retrieve triples that are valid at present time (regarding the Wikidata instance used), 
        set valid_at=wikidata.time_date (the timestamp of the used Wikidata instance) or valid_at="present".

        If a list is provided for subject for instance, it means "Find all the triple with one of the mentioned subject" 
        (there is an OR operator between subjects)

        Args:
            subject (Entity | Iterable[Entity]): Subject(s) to look for
            relation (Relation | Iterable[Relation]) : Relation(s) to look for
            object (Entity | Iterable[Entity]) : Object(s) to look for 
            valid_at (Interval | Date | Iterable[Interval | Date]) : Validity period(s) to look for. 
            - If it's a Date, we look for triples that are valid at this date
            - If it's an Interval, we look for exact validity period match
        """
        super().__init__(subject, relation, object)
        self.valid_at = valid_at

    @staticmethod
    def from_triple(triple: TimedTriple | Triple) -> TripleQuery | TimedTripleQuery:
        if type(triple) is Triple:
            return TripleQuery.from_triple(triple)
        elif type(triple) is TimedTriple:
            return TimedTripleQuery(triple.subject, triple.relation, triple.object, triple.valid_between)
        else:
            raise TypeError('"triple" must be a Triple or a TimedTriple')

    def __repr__(self) -> str:
        triple_query = super().__repr__()
        vb = self.valid_between
        vb = "?" if vb is None else vb
        return triple_query[:-1] + ', valid_between=%s)' % vb
    

def _gen_exact_match_query(prop : Proposition | Iterable[Proposition]) -> Query:
    """Generate a query that look for the specific given proposition inside a knowledge base

    Args:
        prop (Proposition): Proposition to find

    Raises:
        Exception: In case the exact match query could not be generated

    Returns:
        Query: Exact match query
    """
    if type(prop) is Triple:
        return TripleQuery(prop.subject, prop.relation, prop.object)
    elif type(prop) is TimedTriple:
        return TimedTripleQuery(prop.subject, prop.relation, prop.object, prop.valid_between)
    raise Exception('Exact match query of the given proposition could not be generated !')

def _gen_empty_query(prop_cls : type) -> Query:
    """Generate a query with no condition that matches with every proposition in the knowledge base

    Args:
        prop_cls (type): Proposition class

    Returns:
        Query
    """
    if prop_cls is Triple:
        return TripleQuery(None, None, None)
    elif prop_cls is TimedTriple:
        return TimedTripleQuery(None, None, None, None)
    raise Exception('Empty query of the given proposition class could not be generated !')

class KnowledgeBase(metaclass=ABCMeta):
    """An abstract class for a Knowledge graph which is simply a collection of triples.
    """
    PROPOSITION_TYPE = Proposition
    @abstractmethod
    def find(self, query : Query) -> Iterable[Proposition]:
        """Retrieve propositions that satisfy the given query

        Args:
            query (Query): Query

        Returns:
            Iterable[Proposition]: An iterable of proposition
        """
        pass

    def contains(self, prop : Proposition | Iterable[Proposition]) -> bool:
        """Check if the knowledge base contains the given proposition

        Args:
            prop (Proposition) : Proposition to find

        Returns:
            bool: True if contains, else False
        """
        it = self.find(_gen_exact_match_query(prop))
        try:
            next(iter(it))
            return True
        except StopIteration:
            return False
        

    def iterate(self) -> Iterable[Proposition]:
        """Iterate through all proposition in the knowledge base

        Returns:
            Iterable[Proposition]: Iterator of propositions
        """
        return self.find(_gen_empty_query(self.__class__.PROPOSITION_TYPE))
    
    def sample_propositions(self, n : int) -> Iterable[Proposition]:
        """Randomly sample propositions from this knowledge base

        Args:
            n (int): Sample size
        """
        raise NotImplementedError
    
    def sample_subjects(self, n : int) -> Iterable[Entity]:
        """Randomly sample entities from this knowledge base

        Args:
            n (int): Sample size
        """
        raise NotImplementedError
    
    def sample_relations(self, n : int) -> Iterable[Relation]:
        """Randomly sample relations from this knowledge base

        Args:
            n (int): Sample size
        """
        raise NotImplementedError
    
    def __iter__(self) -> Iterable[Proposition]:
        return self.iterate()
    
    def inject_info(self, entities : Iterable[Entity]) -> None:
        """Inject info (label and description) to entities when possible. The injection is inplace

        Args:
            entities (Iterable[Entity]): Iterable of entities
        """
        raise NotImplementedError