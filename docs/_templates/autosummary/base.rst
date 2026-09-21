{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{% if module + "." + objname in gallery_backreferences %}
.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Examples using ``{{ objname }}``
{% endif %}
