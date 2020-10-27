class EnumStr(str):
    def __str__(self):
        return "%s" % (self._name_)

    def lower(self):
        return '{}'.format(self._name_).lower()

    def upper(self):
        return '{}'.format(self._name_).upper()
